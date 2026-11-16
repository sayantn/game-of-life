use std::iter::{FusedIterator, zip};
use std::mem;

use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use crate::{
    Backend, Batch, Buffer, BufferOf, BytesOf, Chunk, ChunkOf, ConfigOf, DefaultBackend,
    NeighborData, Owned, OwnedOf,
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

    batch: Batch<'static, B::Buffer>,
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
        mut f: impl FnMut(usize, usize) -> B::Chunk,
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

        let chunk_values: Vec<_> = (0..v_chunks)
            .flat_map(|i| (0..h_chunks).map(move |j| (i, j)))
            .map(|(i, j)| f(j, i))
            .collect();

        // SAFETY: Every handle in `chunks` was just minted by `buffer.grow_exact`.
        unsafe { buffer.store_batched(&mut chunks, &chunk_values) };

        Self {
            buffer,
            chunks: chunks.into_boxed_slice(),
            alternate: alternate.into_boxed_slice(),
            topology,
            batch: Batch::new(),
            v_chunks,
            h_chunks,
        }
    }

    /// Constructs a `h_chunks × v_chunks` toroidal world filled with uniformly random cells.
    #[inline]
    pub fn random(config: ConfigOf<B>, h_chunks: usize, v_chunks: usize, rng: &mut impl Rng) -> Self
    where
        StandardUniform: Distribution<BytesOf<B>>,
    {
        Self::from_fn(config, h_chunks, v_chunks, |_, _| B::Chunk::random(rng))
    }

    /// Constructs a `h_chunks × v_chunks` toroidal world with all cells initially dead.
    #[inline]
    pub fn empty(config: ConfigOf<B>, h_chunks: usize, v_chunks: usize) -> Self {
        Self::from_fn(config, h_chunks, v_chunks, |_, _| Chunk::empty())
    }

    #[inline]
    pub fn h_chunks(&self) -> usize {
        self.h_chunks
    }

    #[inline]
    pub fn v_chunks(&self) -> usize {
        self.v_chunks
    }

    /// Returns the total number of cells across all chunks (`h_chunks * v_chunks * SIZE²`).
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.chunks.len() * ChunkOf::<B>::SIZE * ChunkOf::<B>::SIZE
    }

    /// Returns the total number of live cells currently in the world.
    #[inline]
    pub fn num_alive(&self) -> usize {
        self.chunks_iter().map(|chunk| chunk.num_alive()).sum()
    }

    /// Iterates over all chunks in row-major order, downloading each chunk from the backend.
    #[inline]
    pub fn chunks_iter(
        &self,
    ) -> impl ExactSizeIterator<Item = B::Chunk> + DoubleEndedIterator + FusedIterator {
        self.chunks.iter().map(|chunk| {
            // SAFETY: Every `Owned` handle in `self.chunks` was minted by `self.buffer`.
            unsafe { self.buffer.load(chunk.id()) }
        })
    }

    /// Advances the entire toroidal grid by one generation.
    #[inline]
    pub fn clock(&mut self) {
        let mut batch = mem::take(&mut self.batch).recycle();

        batch.reserve_alive(self.chunks.len());
        for (chunk, links) in self.chunks.iter().zip(&self.topology) {
            let nbhd = links.map(|idx| {
                // SAFETY: `self.topology` only contains indices in `0..self.chunks.len()`.
                unsafe { self.chunks.get_unchecked(idx).id() }
            });
            batch.push_alive_within_capacity(chunk.id(), nbhd);
        }

        // SAFETY: All IDs in `batch` come from `self.chunks` and all destination handles in
        // `self.alternate` are disjoint `Owned` tokens minted by `self.buffer`.
        unsafe {
            self.buffer
                .step_batched(&batch, &mut self.alternate, &mut []);
        }

        self.batch = batch.recycle();

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
            unsafe { new_buffer.copy(dst, &self.buffer, src.id()) };
        }

        Self {
            buffer: new_buffer,
            chunks: new_chunks.into_boxed_slice(),
            alternate: alternate.into_boxed_slice(),
            topology: self.topology.clone(),
            batch: Batch::default(),
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
            unsafe { self.buffer.copy(dst, &source.buffer, src.id()) };
        }
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
