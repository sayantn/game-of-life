use std::iter::FusedIterator;
use std::mem;

use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use crate::primitives::NeighborData;
use crate::traits::{Backend, Batch, Buffer, Chunk, Owned};

pub struct ToroidalWorld<B: Backend> {
    buffer: B::Buffer,
    chunks: Box<[<B::Buffer as Buffer>::Owned]>,
    alternate: Box<[<B::Buffer as Buffer>::Owned]>,
    topology: Box<[NeighborData<usize>]>,

    batch: Batch<'static, B::Buffer>,
    results: Vec<B::Activity>,
    // v_chunks: usize,
    // h_chunks: usize,
}

impl<B: Backend> ToroidalWorld<B> {
    pub fn from_fn(
        config: <B::Buffer as Buffer>::Config,
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

        let mut buffer = <B::Buffer as Buffer>::new(config);
        buffer.grow(2 * total_chunks, &mut chunks);

        chunks.truncate(2 * total_chunks);
        let alternate = chunks.split_off(total_chunks);

        let chunk_values: Vec<_> = (0..v_chunks)
            .flat_map(|i| (0..h_chunks).map(move |j| (i, j)))
            .map(|(i, j)| f(i, j))
            .collect();

        unsafe { buffer.store_batched(&mut chunks, &chunk_values) };

        Self {
            buffer,
            chunks: chunks.into_boxed_slice(),
            alternate: alternate.into_boxed_slice(),
            topology,
            batch: Batch::new(),
            results: Vec::new(),
            // v_chunks,
            // h_chunks,
        }
    }

    pub fn random(
        config: <B::Buffer as Buffer>::Config,
        h_chunks: usize,
        v_chunks: usize,
        rng: &mut impl Rng,
    ) -> Self
    where
        StandardUniform: Distribution<<B::Chunk as Chunk>::Bytes>,
    {
        Self::from_fn(config, h_chunks, v_chunks, |_, _| B::Chunk::random(rng))
    }

    pub fn empty(config: <B::Buffer as Buffer>::Config, h_chunks: usize, v_chunks: usize) -> Self {
        Self::from_fn(config, h_chunks, v_chunks, |_, _| Chunk::empty())
    }

    pub fn num_cells(&self) -> usize {
        self.chunks.len() * <B::Chunk as Chunk>::SIZE * <B::Chunk as Chunk>::SIZE
    }

    pub fn num_alive(&self) -> usize {
        self.chunks_iter().map(|chunk| chunk.num_alive()).sum()
    }

    pub fn chunks_iter(
        &self,
    ) -> impl ExactSizeIterator<Item = B::Chunk> + DoubleEndedIterator + FusedIterator {
        self.chunks
            .iter()
            .map(|chunk| unsafe { self.buffer.load(chunk.id()) })
    }

    pub fn clock(&mut self) {
        let mut batch = mem::take(&mut self.batch).recycle();

        batch.reserve_alive(self.chunks.len());
        for (chunk, links) in self.chunks.iter().zip(&self.topology) {
            let nbhd = links.map(|idx| unsafe { self.chunks.get_unchecked(idx).id() });
            batch.push_alive_within_capacity(chunk.id(), nbhd);
        }

        self.results.clear();

        unsafe {
            self.buffer.compute_batched(
                &batch,
                &mut self.alternate,
                &mut Vec::new(),
                &mut self.results,
            );
        }

        self.batch = batch.recycle();

        mem::swap(&mut self.chunks, &mut self.alternate);
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
