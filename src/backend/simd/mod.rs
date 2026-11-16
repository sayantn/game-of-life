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

/// SAFETY:
/// `SimdBuffer` exclusively owns its heap allocation (`Vec`). The underlying data
/// (`SimdChunk`) consists purely of SIMD vector integers (Plain Old Data) and contains
/// no thread-local state, raw pointers, or un-sendable types. Therefore, it is
/// perfectly safe to transfer the entire buffer across thread boundaries.
unsafe impl Send for SimdBuffer {}

/// SAFETY:
/// `SimdBuffer` utilizes interior mutability (e.g., `UnsafeCell` or raw pointers) to
/// allow disjoint mutations across chunks. This is safe to share across threads (`Sync`)
/// because the capability system (`Borrowed` and `BorrowedMut`) mathematically guarantees
/// at compile time that no two threads can ever acquire overlapping mutable references,
/// and no thread can read a chunk while another is writing to it. The capability tokens
/// act as a zero-cost compile-time lock.
unsafe impl Sync for SimdBuffer {}

impl SimdBuffer {
    /// Retrieves a shared reference to the chunk at the given handle index.
    ///
    /// # Safety
    /// - `id` must have been minted by this exact `SimdBuffer`.
    /// - The `Borrowed` capability token statically proves that no exclusive (`BorrowedMut`)
    ///   token exists for this slot for the duration of `'a`. Therefore, no mutable aliasing
    ///   or data races can occur while reading.
    #[inline]
    unsafe fn get(&self, id: SimdHandle) -> &SimdChunk {
        // SAFETY: Caller guarantees `id` was minted by `self` (so `id.0 < self.chunks.len()`)
        // and no mutable reference to slot `id` exists while `id` is live.
        unsafe { self.chunks.get_unchecked(id).as_ref_unchecked() }
    }

    /// Retrieves an exclusive mutable reference to the chunk at the given handle index.
    ///
    /// # Safety
    /// - `id` must have been minted by this exact `SimdBuffer`.
    /// - The caller's possession of a `BorrowedMut` capability mathematically guarantees
    ///   at the type level that no other tokens (shared or exclusive) exist for this
    ///   specific slot. Therefore, mutating through the derived pointer is strictly
    ///   disjoint and free of aliasing.
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

/// SAFETY:
/// - **Handle Exclusivity:** `grow` and `grow_exact` increase the capacity of the backing
///   storage and mint fresh, unique `Owned` capability tokens strictly for the newly allocated
///   range `old_cap..new_cap`. Existing handles are never duplicated.
/// - **Stability Across Reallocation:** Capability tokens wrap a purely logical `usize` index.
///   When the backing storage resizes and moves on the heap, the raw pointers change, but the
///   logical indices within the capability tokens remain universally valid.
/// - **Interior Mutability & Data Races:** The capability system (`Borrowed` / `BorrowedMut`)
///   acts as a compile-time lock over the raw slots. Because `step_batched` strictly consumes
///   `TickBatch` jobs containing disjoint `BorrowedMut` destination tokens, parallel threads
///   (via Rayon) can safely execute unaliased concurrent writes to the underlying memory.
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
