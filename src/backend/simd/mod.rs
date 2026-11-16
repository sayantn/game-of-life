use std::cell::SyncUnsafeCell;
use std::marker::PhantomData;

mod chunk;
mod row;

pub use chunk::SimdActivity;
pub use chunk::SimdChunk;

use crate::{Backend, Batch, Buffer, Chunk, Owned};

/// Single-node, AVX2-accelerated CPU compute backend.
pub struct SimdBackend;

impl Backend for SimdBackend {
    type Chunk = SimdChunk;
    type Buffer = SimdBuffer;
    type Activity = SimdActivity;
}

/// Exclusive ownership handle for a single slot in [`SimdBuffer`].
///
/// Intentionally does not implement `Clone` or `Copy`.
#[derive(Debug)]
#[repr(transparent)]
pub struct SimdOwned(usize);

impl Owned for SimdOwned {
    type Id<'a> = SimdId<'a>;

    #[inline]
    fn id(&self) -> Self::Id<'_> {
        SimdId(self.0, PhantomData)
    }
}

/// Lightweight, `Copy` read-only capability handle borrowing a slot in [`SimdBuffer`]
/// for lifetime `'a`.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct SimdId<'a>(usize, PhantomData<&'a SimdOwned>);

/// Host-memory slab backing [`SimdBackend`], storing chunks in contiguous `SyncUnsafeCell`s
/// for lock-free parallel reads and disjoint writes.
pub struct SimdBuffer {
    chunks: Vec<SyncUnsafeCell<SimdChunk>>,
}

impl SimdBuffer {
    #[inline]
    unsafe fn get<'a>(&'a self, id: SimdId<'a>) -> &'a SimdChunk {
        // SAFETY: Caller guarantees `id` was minted by `self` (so `id.0 < self.chunks.len()`)
        // and no mutable reference to slot `id.0` exists while `id` is live.
        unsafe { &*self.chunks.get_unchecked(id.0).get() }
    }

    #[inline]
    unsafe fn get_mut<'a>(&'a self, owned: &'a mut SimdOwned) -> &'a mut SimdChunk {
        // SAFETY: Caller guarantees `owned` was minted by `self` (so `owned.0 < self.chunks.len()`)
        // and `&mut SimdOwned` guarantees exclusive access to slot `owned.0`.
        unsafe { &mut *self.chunks.get_unchecked(owned.0).get() }
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
    type Chunk = SimdChunk;
    type Activity = SimdActivity;
    type Owned = SimdOwned;
    type Id<'a> = SimdId<'a>;

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
    fn grow(&mut self, additional: usize, out: &mut Vec<Self::Owned>) {
        if additional == 0 {
            return;
        }

        let old_cap = self.chunks.len();
        self.chunks.reserve(additional);
        // Expand up to the full amortized `Vec` capacity so subsequent growths are rare.
        let new_cap = self.chunks.capacity();

        self.chunks
            .resize_with(new_cap, || SyncUnsafeCell::new(SimdChunk::empty()));

        // Push in reverse so `Vec::pop` hands out sequential indices `old_cap..new_cap`.
        out.reserve(new_cap - old_cap);
        out.extend((old_cap..new_cap).rev().map(SimdOwned));
    }

    #[inline]
    fn grow_exact(&mut self, additional: usize, out: &mut Vec<Self::Owned>) {
        if additional == 0 {
            return;
        }

        let old_cap = self.chunks.len();
        self.chunks.reserve_exact(additional);
        let new_cap = self.chunks.capacity();

        self.chunks
            .resize_with(new_cap, || SyncUnsafeCell::new(SimdChunk::empty()));

        // Push in reverse so `Vec::pop` hands out sequential indices `old_cap..new_cap`.
        out.reserve(new_cap - old_cap);
        out.extend((old_cap..new_cap).rev().map(SimdOwned));
    }

    #[inline]
    unsafe fn load_to(&self, id: Self::Id<'_>, dst: &mut Self::Chunk) {
        // SAFETY: Caller guarantees `id` was minted by `self`.
        unsafe { self.get(id).clone_into(dst) };
    }

    #[inline]
    unsafe fn store(&self, dst: &mut Self::Owned, src: &Self::Chunk) {
        // SAFETY: Caller guarantees `dst` was minted by `self`.
        unsafe { src.clone_into(self.get_mut(dst)) };
    }

    #[inline]
    unsafe fn copy(&self, dst: &mut Self::Owned, from: &Self, src: Self::Id<'_>) {
        // SAFETY: Caller guarantees `dst` was minted by `self` and `src` was minted by `from`.
        unsafe { from.get(src).clone_into(self.get_mut(dst)) };
    }

    #[inline]
    unsafe fn compute_batched(
        &self,
        batch: &Batch<'_, Self>,
        alive_dsts: &mut [Self::Owned],
        dead_dsts: &mut [Self::Owned],
        out: &mut Vec<SimdActivity>,
    ) {
        assert_eq!(alive_dsts.len(), batch.alive_len());
        assert_eq!(dead_dsts.len(), batch.dead_len());

        let total_len = alive_dsts.len() + dead_dsts.len();

        // SAFETY: Caller guarantees all source/neighbor IDs and destination `Owned` handles were
        // minted by `self`, and the capability rules ensure no destination slot aliases any input
        // or other destination slot.

        if cfg!(feature = "parallel") && total_len >= 32 {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;

                let alive_results = batch
                    .alive_srcs()
                    .par_iter()
                    .zip(batch.alive_nbhds())
                    .zip(alive_dsts)
                    .map(|((&src, &neighbors), dst)| unsafe {
                        chunk::apply_rules(
                            self.get(src),
                            neighbors.map(|idx| self.get(idx)),
                            self.get_mut(dst),
                        )
                    });

                let dead_results =
                    batch
                        .dead_nbhds()
                        .par_iter()
                        .zip(dead_dsts)
                        .map(|(&neighbors, dst)| unsafe {
                            chunk::check_activity(
                                neighbors.map(|idx| self.get(idx)),
                                self.get_mut(dst),
                            )
                        });

                alive_results.chain(dead_results).collect_into_vec(out);
            }
        } else {
            out.clear();

            out.extend(
                batch
                    .alive_srcs()
                    .iter()
                    .zip(batch.alive_nbhds())
                    .zip(alive_dsts)
                    .map(|((&src, &neighbors), dst)| unsafe {
                        chunk::apply_rules(
                            self.get(src),
                            neighbors.map(|idx| self.get(idx)),
                            self.get_mut(dst),
                        )
                    }),
            );

            out.extend(
                batch
                    .dead_nbhds()
                    .iter()
                    .zip(dead_dsts)
                    .map(|(&neighbors, dst)| unsafe {
                        chunk::check_activity(neighbors.map(|idx| self.get(idx)), self.get_mut(dst))
                    }),
            );
        }
    }

    #[inline]
    unsafe fn step_batched(
        &self,
        batch: &Batch<'_, Self>,
        alive_dsts: &mut [Self::Owned],
        dead_dsts: &mut [Self::Owned],
    ) {
        assert_eq!(alive_dsts.len(), batch.alive_len());
        assert_eq!(dead_dsts.len(), batch.dead_len());

        let total_len = alive_dsts.len() + dead_dsts.len();

        // SAFETY: Caller guarantees all source/neighbor IDs and destination `Owned` handles were
        // minted by `self`, and the capability rules ensure no destination slot aliases any input
        // or other destination slot.

        if cfg!(feature = "parallel") && total_len >= 32 {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;

                let alive_results = batch
                    .alive_srcs()
                    .par_iter()
                    .zip(batch.alive_nbhds())
                    .zip(alive_dsts)
                    .map(|((&src, &neighbors), dst)| unsafe {
                        chunk::apply_rules(
                            self.get(src),
                            neighbors.map(|idx| self.get(idx)),
                            self.get_mut(dst),
                        );
                    });

                let dead_results =
                    batch
                        .dead_nbhds()
                        .par_iter()
                        .zip(dead_dsts)
                        .map(|(&neighbors, dst)| unsafe {
                            chunk::check_activity(
                                neighbors.map(|idx| self.get(idx)),
                                self.get_mut(dst),
                            );
                        });

                alive_results.chain(dead_results).for_each(drop);
            }
        } else {
            batch
                .alive_srcs()
                .iter()
                .zip(batch.alive_nbhds())
                .zip(alive_dsts)
                .for_each(|((&src, &neighbors), dst)| unsafe {
                    chunk::apply_rules(
                        self.get(src),
                        neighbors.map(|idx| self.get(idx)),
                        self.get_mut(dst),
                    );
                });

            batch
                .dead_nbhds()
                .iter()
                .zip(dead_dsts)
                .for_each(|(&neighbors, dst)| unsafe {
                    chunk::check_activity(neighbors.map(|idx| self.get(idx)), self.get_mut(dst));
                });
        }
    }
}
