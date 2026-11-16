use std::cell::UnsafeCell;

mod chunk;
mod row;

pub use chunk::SimdActivity;
use chunk::SimdChunk;

use crate::{
    Backend, Borrowed, BorrowedMut, Buffer, Chunk, ComputeJob, Owned, TickBatch, VacantComputeJob,
};

/// Single-node, AVX2-accelerated CPU compute backend.
#[derive(Debug, Clone, Copy)]
pub struct SimdBackend;

impl Backend for SimdBackend {
    type Buffer = SimdBuffer;
    type Activity = SimdActivity;
}

pub type SimdHandle = usize;

/// Host-memory slab backing [`SimdBackend`], storing chunks in contiguous `SyncUnsafeCell`s
/// for lock-free parallel reads and disjoint writes.
#[derive(Debug)]
pub struct SimdBuffer {
    chunks: Vec<UnsafeCell<SimdChunk>>,
}

unsafe impl Sync for SimdBuffer {}

impl SimdBuffer {
    /// Retrieves a shared reference to the chunk at the given handle index.
    ///
    /// # Safety
    /// - `id` must have been minted by this exact `SimdBuffer` instance.
    /// - The caller must ensure that no exclusive (`&mut`) reference to the slot
    ///   exists for the duration of the returned lifetime.
    #[inline]
    unsafe fn get(&self, id: SimdHandle) -> &SimdChunk {
        // SAFETY: Caller guarantees `id` was minted by `self` (so `id.0 < self.chunks.len()`)
        // and no mutable reference to slot `id` exists while `id` is live.
        unsafe { self.chunks.get_unchecked(id).as_ref_unchecked() }
    }

    /// Retrieves an exclusive reference to the chunk at the given handle index.
    ///
    /// # Safety
    /// - `id` must have been minted by this exact `SimdBuffer` instance.
    /// - The caller must ensure exclusive access to the slot; no other references
    ///   (shared or exclusive) to this slot may exist for the duration of the returned lifetime.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self, id: SimdHandle) -> &mut SimdChunk {
        // SAFETY: Caller guarantees `owned` was minted by `self` (so `owned.0 < self.chunks.len()`)
        // and `&mut SimdOwned` guarantees exclusive access to slot `owned.0`.
        unsafe { self.chunks.get_unchecked(id).as_mut_unchecked() }
    }
}

impl Default for SimdBuffer {
    #[inline]
    fn default() -> Self {
        Self::new(())
    }
}

unsafe impl Buffer for SimdBuffer {
    type Config = ();
    type Activity = SimdActivity;
    type Handle = SimdHandle;

    #[inline]
    fn new((): Self::Config) -> Self {
        Self { chunks: Vec::new() }
    }

    #[inline]
    fn config(&self) -> &Self::Config {
        &()
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.chunks.len()
    }

    #[inline]
    fn grow(&mut self, additional: usize, out: &mut Vec<Owned<SimdHandle>>) {
        if additional == 0 {
            return;
        }

        let old_cap = self.chunks.len();
        self.chunks.reserve(additional);
        // Expand up to the full amortized `Vec` capacity so subsequent growths are rare.
        let new_cap = self.chunks.capacity();

        self.chunks
            .resize_with(new_cap, || UnsafeCell::new(SimdChunk::empty()));

        // Push in reverse so `Vec::pop` hands out sequential indices `old_cap..new_cap`.
        out.reserve(new_cap - old_cap);
        out.extend((old_cap..new_cap).rev().map(|raw| {
            // SAFETY: `raw` monotonically increases, so the same handle is never minted twice
            unsafe { Owned::from_raw(raw) }
        }));
    }

    #[inline]
    fn grow_exact(&mut self, additional: usize, out: &mut Vec<Owned<SimdHandle>>) {
        if additional == 0 {
            return;
        }

        let old_cap = self.chunks.len();
        self.chunks.reserve_exact(additional);
        let new_cap = self.chunks.capacity();

        self.chunks
            .resize_with(new_cap, || UnsafeCell::new(SimdChunk::empty()));

        // Push in reverse so `Vec::pop` hands out sequential indices `old_cap..new_cap`.
        out.reserve(new_cap - old_cap);
        out.extend((old_cap..new_cap).rev().map(|raw| {
            // SAFETY: `raw` monotonically increases, so the same handle is never minted twice
            unsafe { Owned::from_raw(raw) }
        }));
    }

    #[inline]
    unsafe fn load_to(&self, id: Borrowed<'_, SimdHandle>, dst: &mut Chunk) {
        // SAFETY: Caller guarantees `id` was minted by `self`.
        unsafe { self.get(id.raw()).convert_into(dst) };
    }

    #[inline]
    unsafe fn store(&self, dst: BorrowedMut<'_, SimdHandle>, src: &Chunk) {
        // SAFETY: Caller guarantees `dst` was minted by `self`.
        unsafe { src.convert_into(self.get_mut(dst.raw())) };
    }

    #[inline]
    unsafe fn copy(
        &self,
        dst: BorrowedMut<'_, SimdHandle>,
        from: &Self,
        src: Borrowed<'_, SimdHandle>,
    ) {
        // SAFETY: Caller guarantees `dst` was minted by `self` and `src` was minted by `from`.
        unsafe { from.get(src.raw()).clone_into(self.get_mut(dst.raw())) };
    }

    #[inline]
    unsafe fn step_batched(
        &self,
        jobs: &mut TickBatch<'_, SimdHandle>,
        activities: Option<&mut Vec<SimdActivity>>,
    ) {
        #![allow(clippy::undocumented_unsafe_blocks, clippy::unnecessary_safety_comment)]

        let total_len = jobs.compute.len() + jobs.vacant_compute.len();

        // SAFETY: Caller guarantees all source/neighbor IDs and destination `Owned` handles were
        // minted by `self`, and the capability rules ensure no destination slot aliases any input
        // or other destination slot.

        if cfg!(feature = "parallel") && total_len >= 32 {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;

                let alive_results = jobs.compute.par_iter_mut().map(
                    |ComputeJob {
                         center,
                         neighbors,
                         dst,
                     }| unsafe {
                        chunk::apply_rules(
                            self.get(center.raw()),
                            neighbors.map(|idx| self.get(idx.raw())),
                            self.get_mut(dst.raw()),
                        )
                    },
                );

                let dead_results = jobs.vacant_compute.par_iter_mut().map(
                    |VacantComputeJob { neighbors, dst }| unsafe {
                        chunk::check_activity(
                            neighbors.map(|idx| self.get(idx.raw())),
                            self.get_mut(dst.raw()),
                        )
                    },
                );

                if let Some(output) = activities {
                    output.par_extend(alive_results.chain(dead_results));
                } else {
                    alive_results.chain(dead_results).for_each(drop);
                }
            }
        } else {
            let alive_results = jobs.compute.iter_mut().map(
                |ComputeJob {
                     center,
                     neighbors,
                     dst,
                 }| unsafe {
                    chunk::apply_rules(
                        self.get(center.raw()),
                        neighbors.map(|idx| self.get(idx.raw())),
                        self.get_mut(dst.raw()),
                    )
                },
            );

            let dead_results =
                jobs.vacant_compute
                    .iter_mut()
                    .map(|VacantComputeJob { neighbors, dst }| unsafe {
                        chunk::check_activity(
                            neighbors.map(|idx| self.get(idx.raw())),
                            self.get_mut(dst.raw()),
                        )
                    });

            if let Some(output) = activities {
                output.extend(alive_results.chain(dead_results));
            } else {
                alive_results.chain(dead_results).for_each(drop);
            }
        }
    }
}
