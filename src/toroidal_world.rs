use std::fmt::Debug;
use std::iter::{FusedIterator, zip};
use std::mem;

use rand::Rng;

use crate::helpers::push_within_capacity;
use crate::{
    Backend, Buffer, BufferOf, Chunk, ComputeJob, ConfigOf, DefaultBackend, HandleOf, NeighborData,
    OwnedOf, StepScratch,
};

/// Fixed-size, wrap-around (toroidal) Game of Life simulation grid backed by [`Backend`] `B`.
///
/// Uses double-buffered slot arrays (`chunks` and `alternate`) and precomputed neighbor
/// topology to step all chunks in lockstep with zero per-tick allocations.
pub struct ToroidalWorld<B: Backend = DefaultBackend> {
    buffer: B::Buffer,
    chunks: Box<[OwnedOf<B>]>,
    alternate: Box<[OwnedOf<B>]>,
    topology: Box<[NeighborData<usize>]>,
    jobs: StepScratch<HandleOf<B>>,
    v_chunks: usize,
    h_chunks: usize,
}

impl<B: Backend> ToroidalWorld<B> {
    /// Constructs an `h_chunks × v_chunks` toroidal world, initializing each chunk at
    /// column `x` and row `y` via `f(x, y)`.
    #[inline]
    pub fn from_fn(
        config: ConfigOf<B>,
        h_chunks: usize,
        v_chunks: usize,
        mut f: impl FnMut(usize, usize) -> Chunk,
    ) -> Self {
        let total_chunks = v_chunks * h_chunks;

        let topology = (0..v_chunks)
            .flat_map(|i| (0..h_chunks).map(move |j| (i, j)))
            .map(|(i, j)| {
                let rows = [
                    ((i + v_chunks - 1) % v_chunks) * h_chunks,
                    i * h_chunks,
                    ((i + 1) % v_chunks) * h_chunks,
                ];
                let cols = [(j + h_chunks - 1) % h_chunks, j, (j + 1) % h_chunks];

                NeighborData::from_fn(|dir| {
                    let (dx, dy) = dir.offset();
                    rows[(dy + 1) as usize] + cols[(dx + 1) as usize]
                })
            })
            .collect();

        let mut chunks = Vec::new();

        let mut buffer = BufferOf::<B>::new(config);
        buffer.grow_exact(2 * total_chunks, &mut chunks);

        let alternate = chunks.split_off(total_chunks);

        for ((i, j), dst) in (0..v_chunks)
            .flat_map(|i| (0..h_chunks).map(move |j| (i, j)))
            .zip(&mut chunks)
        {
            // SAFETY: `dst` was just minted by `buffer.grow_exact`.
            unsafe { buffer.store(dst.borrow_mut(), &f(i, j)) };
        }

        Self {
            buffer,
            chunks: chunks.into_boxed_slice(),
            alternate: alternate.into_boxed_slice(),
            topology,
            jobs: StepScratch::new(),
            v_chunks,
            h_chunks,
        }
    }

    /// Constructs a `h_chunks × v_chunks` toroidal world filled with uniformly random cells.
    #[inline]
    pub fn random(
        config: ConfigOf<B>,
        h_chunks: usize,
        v_chunks: usize,
        rng: &mut impl Rng,
    ) -> Self {
        Self::from_fn(config, h_chunks, v_chunks, |_, _| {
            Chunk::from_bytes(rng.random())
        })
    }

    /// Constructs a `h_chunks × v_chunks` toroidal world with all cells initially dead.
    #[inline]
    pub fn empty(config: ConfigOf<B>, h_chunks: usize, v_chunks: usize) -> Self {
        Self::from_fn(config, h_chunks, v_chunks, |_, _| Chunk::empty())
    }

    #[inline]
    pub const fn h_chunks(&self) -> usize {
        self.h_chunks
    }

    #[inline]
    pub const fn v_chunks(&self) -> usize {
        self.v_chunks
    }

    /// Returns the total number of cells across all chunks (`h_chunks * v_chunks * SIZE²`).
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.chunks.len() * Chunk::SIZE * Chunk::SIZE
    }

    /// Returns the total number of live cells currently in the world.
    #[inline]
    pub fn population(&self) -> u32 {
        self.chunks_iter().map(|chunk| chunk.population()).sum()
    }

    /// Iterates over all chunks in row-major order, downloading each chunk from the backend.
    #[inline]
    pub fn chunks_iter(
        &self,
    ) -> impl ExactSizeIterator<Item = Chunk> + DoubleEndedIterator + FusedIterator {
        self.chunks.iter().map(|chunk| {
            // SAFETY: Every `Owned` handle in `self.chunks` was minted by `self.buffer`.
            unsafe { self.buffer.load(chunk.borrow()) }
        })
    }

    /// Advances the entire toroidal grid by one generation.
    #[inline]
    pub fn clock(&mut self) {
        let mut jobs = self.jobs.start_tick();

        jobs.compute.reserve(self.chunks.len());
        for ((chunk, links), dst) in zip(&self.chunks, &self.topology).zip(&mut self.alternate) {
            let neighbors = links.map(|idx| {
                // SAFETY: `self.topology` only contains indices in `0..self.chunks.len()`.
                unsafe { self.chunks.get_unchecked(idx).borrow() }
            });
            push_within_capacity(
                &mut jobs.compute,
                ComputeJob {
                    neighbors,
                    center: chunk.borrow(),
                    dst: dst.borrow_mut(),
                },
            );
        }

        // SAFETY: All IDs in `batch` come from `self.chunks` and all destination handles in
        // `self.alternate` are disjoint `Owned` tokens minted by `self.buffer`.
        unsafe {
            self.buffer.step_batched(&mut jobs, None);
        }

        self.jobs.finish_tick(jobs);

        mem::swap(&mut self.chunks, &mut self.alternate);
    }
}

impl<B: Backend> Clone for ToroidalWorld<B> {
    #[inline]
    fn clone(&self) -> Self {
        let mut new_buffer = BufferOf::<B>::new(self.buffer.config().clone());
        let mut new_chunks = Vec::with_capacity(2 * self.chunks.len());
        new_buffer.grow_exact(2 * self.chunks.len(), &mut new_chunks);

        let alternate = new_chunks.split_off(self.chunks.len());

        for (dst, src) in zip(&mut new_chunks, &self.chunks) {
            // SAFETY: `dst` was minted by `new_buffer` and `src` was minted by `self.buffer`.
            unsafe { new_buffer.copy(dst.borrow_mut(), &self.buffer, src.borrow()) };
        }

        Self {
            buffer: new_buffer,
            chunks: new_chunks.into_boxed_slice(),
            alternate: alternate.into_boxed_slice(),
            topology: self.topology.clone(),
            jobs: StepScratch::new(),
            v_chunks: self.v_chunks,
            h_chunks: self.h_chunks,
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        assert_eq!(self.v_chunks, source.v_chunks);
        assert_eq!(self.h_chunks, source.h_chunks);

        for (dst, src) in zip(&mut self.chunks, &source.chunks) {
            // SAFETY: `dst` was minted by `self.buffer` and `src` was minted by `source.buffer`.
            unsafe {
                self.buffer
                    .copy(dst.borrow_mut(), &source.buffer, src.borrow());
            }
        }
    }
}

impl<B: Backend> Debug for ToroidalWorld<B>
where
    BufferOf<B>: Debug,
    OwnedOf<B>: Debug,
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToroidalWorld")
            .field("buffer", &self.buffer)
            .field("chunks", &self.chunks)
            .field("v_chunks", &self.v_chunks)
            .field("h_chunks", &self.h_chunks)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    use crate::backend::SimdBackend;
    use crate::naive::chunk::NaiveChunk as OldChunk;
    use crate::primitives::Dir;

    #[test]
    fn test_toroidal_world_matches_naive() {
        let (h, v) = (2, 2);
        let mut rng = StdRng::seed_from_u64(42);

        let mut world = ToroidalWorld::<SimdBackend>::random((), h, v, &mut rng);

        // Construct the matching naive world
        let mut old_chunks: Vec<OldChunk> = world
            .chunks_iter()
            .map(|c| OldChunk::from_bytes(c.to_bytes()))
            .collect();

        for i in 1..=50 {
            world.clock();

            // Step the naive world
            let mut next_old = old_chunks.clone();

            for i in 0..v {
                let rows = [(i + v - 1) % v, i, (i + 1) % v];
                for j in 0..h {
                    let cols = [(j + h - 1) % h, j, (j + 1) % h];

                    let get_old = |dir: Dir| {
                        let (dx, dy) = dir.offset();
                        &old_chunks[rows[(dy + 1) as usize] * h + cols[(dx + 1) as usize]]
                    };

                    next_old[i * h + j].check(
                        get_old(Dir::N),
                        get_old(Dir::NE),
                        get_old(Dir::E),
                        get_old(Dir::SE),
                        get_old(Dir::S),
                        get_old(Dir::SW),
                        get_old(Dir::W),
                        get_old(Dir::NW),
                    );
                }
            }

            for chunk in &mut next_old {
                chunk.update();
            }
            old_chunks = next_old;

            // Compare the two worlds
            for (idx, new_chunk) in world.chunks_iter().enumerate() {
                let old_from_new = OldChunk::from_bytes(new_chunk.to_bytes());
                assert_eq!(
                    old_chunks[idx],
                    old_from_new,
                    "Mismatch at generation {i}, chunk {idx} (row {}, col {})",
                    idx / h,
                    idx % h
                );
            }
        }
    }
}
