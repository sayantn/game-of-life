use std::cell::UnsafeCell;
use std::marker::PhantomData;

use crate::chunk::{Activity, Chunk, NeighborData};

pub(crate) struct ChunkRef(usize);

#[derive(Clone, Copy)]
pub(crate) struct Id<'a>(usize, PhantomData<&'a ChunkRef>);

impl ChunkRef {
    pub(crate) fn read(&self) -> Id<'_> {
        Id(self.0, PhantomData)
    }
}

pub(crate) struct Arena {
    chunks: Vec<UnsafeCell<Chunk>>,
    free: Vec<usize>,
}

impl Arena {
    pub(crate) fn new() -> Self {
        Self {
            chunks: Vec::new(),
            free: Vec::new(),
        }
    }

    pub(crate) fn alloc(&mut self) -> ChunkRef {
        ChunkRef(self.free.pop().unwrap_or_else(|| {
            self.chunks.push(UnsafeCell::new(Chunk::empty()));
            self.chunks.len() - 1
        }))
    }

    pub(crate) fn alloc_zeroed(&mut self) -> ChunkRef {
        self.own(Chunk::empty())
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn dealloc(&mut self, idx: ChunkRef) {
        self.free.push(idx.0);
    }

    fn get<'a>(&'a self, idx: Id<'a>) -> &'a Chunk {
        unsafe { self.chunks.get_unchecked(idx.0).as_ref_unchecked() }
    }

    fn get_mut<'a>(&'a self, idx: &'a mut ChunkRef) -> &'a mut Chunk {
        unsafe { self.chunks.get_unchecked(idx.0).as_mut_unchecked() }
    }

    pub(crate) fn load<'a>(&'a self, idx: Id<'a>) -> Chunk {
        self.get(idx).clone()
    }

    pub(crate) fn load_to<'a>(&'a self, idx: Id<'a>, dst: &mut Chunk) {
        self.get(idx).clone_into(dst);
    }

    pub(crate) fn store<'a>(&'a self, idx: &'a mut ChunkRef, chunk: Chunk) {
        *self.get_mut(idx) = chunk;
    }

    pub(crate) fn own(&mut self, chunk: Chunk) -> ChunkRef {
        if let Some(i) = self.free.pop() {
            *self.chunks[i].get_mut() = chunk;
            ChunkRef(i)
        } else {
            self.chunks.push(UnsafeCell::new(chunk));
            ChunkRef(self.chunks.len() - 1)
        }
    }

    pub(crate) fn copy<'a, 'b>(&'a self, dst: &'a mut ChunkRef, other: &'b Arena, src: Id<'b>) {
        other.get(src).clone_into(self.get_mut(dst));
    }

    pub(crate) fn compute_batched(
        &self,
        batch: &Batch<'_>,
        alive_dsts: &mut [ChunkRef],
        dead_dsts: &mut [ChunkRef],
        results: &mut Vec<Activity>,
    ) {
        let total_len = alive_dsts.len() + dead_dsts.len();

        if cfg!(feature = "rayon") && total_len >= 128 {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;

                let alive_results = batch
                    .alive_srcs
                    .par_iter()
                    .zip(&batch.alive_nbhds)
                    .zip(alive_dsts)
                    .map(|((&src, &neighbors), dst)| {
                        self.get(src)
                            .apply_rules(neighbors.map(|idx| self.get(idx)), self.get_mut(dst))
                    });

                let dead_results =
                    batch
                        .dead_nbhds
                        .par_iter()
                        .zip(dead_dsts)
                        .map(|(&neighbors, dst)| {
                            Chunk::check_activity(
                                neighbors.map(|idx| self.get(idx)),
                                self.get_mut(dst),
                            )
                        });

                alive_results.chain(dead_results).collect_into_vec(results);
            }
        } else {
            results.extend(
                batch
                    .alive_srcs
                    .iter()
                    .zip(&batch.alive_nbhds)
                    .zip(alive_dsts)
                    .map(|((&src, &neighbors), dst)| {
                        self.get(src)
                            .apply_rules(neighbors.map(|idx| self.get(idx)), self.get_mut(dst))
                    }),
            );

            results.extend(
                batch
                    .dead_nbhds
                    .iter()
                    .zip(dead_dsts)
                    .map(|(&neighbors, dst)| {
                        Chunk::check_activity(neighbors.map(|idx| self.get(idx)), self.get_mut(dst))
                    }),
            );
        }
    }
}

unsafe impl Sync for Arena {}
unsafe impl Send for Arena {}

#[derive(Default)]
pub(crate) struct Batch<'a> {
    alive_srcs: Vec<Id<'a>>,
    alive_nbhds: Vec<NeighborData<Id<'a>>>,
    dead_nbhds: Vec<NeighborData<Id<'a>>>,
}

impl<'a> Batch<'a> {
    pub(crate) fn reserve_alive(&mut self, additional: usize) {
        self.alive_srcs.reserve(additional);
        self.alive_nbhds.reserve(additional);
    }

    pub(crate) fn push_alive_within_capacity(&mut self, src: Id<'a>, nbhd: NeighborData<Id<'a>>) {
        let _ = self.alive_srcs.push_within_capacity(src);
        let _ = self.alive_nbhds.push_within_capacity(nbhd);
    }

    pub(crate) fn push_dead(&mut self, nbhd: NeighborData<Id<'a>>) {
        self.dead_nbhds.push(nbhd);
    }

    pub(crate) fn reinterpret<'b>(self) -> Batch<'b> {
        unsafe {
            Batch {
                alive_srcs: recycle_vec(self.alive_srcs),
                alive_nbhds: recycle_vec(self.alive_nbhds),
                dead_nbhds: recycle_vec(self.dead_nbhds),
            }
        }
    }
}

unsafe fn recycle_vec<T, U>(mut a: Vec<T>) -> Vec<U> {
    a.clear();
    const {
        assert!(size_of::<T>() == size_of::<U>());
        assert!(align_of::<T>() == align_of::<U>());
    };

    let (ptr, len, cap) = a.into_raw_parts();
    unsafe { Vec::from_raw_parts(ptr.cast(), len, cap) }
}
