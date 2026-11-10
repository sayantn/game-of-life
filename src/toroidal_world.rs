use std::mem;

use rand::Rng;

use crate::chunk::{self, Chunk, NeighborData};

#[derive(Clone)]
pub struct ToroidalWorld {
    chunks: Box<[Chunk]>,
    alternate: Box<[Chunk]>,
    topology: Box<[NeighborData<usize>]>,
    // v_chunks: usize,
    // h_chunks: usize,
}

impl ToroidalWorld {
    pub fn from_fn(
        h_chunks: usize,
        v_chunks: usize,
        mut f: impl FnMut(usize, usize) -> Chunk,
    ) -> Self {
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

        Self {
            chunks: (0..v_chunks)
                .flat_map(|i| (0..h_chunks).map(move |j| (i, j)))
                .map(|(i, j)| f(i, j))
                .collect(),
            alternate: (0..v_chunks * h_chunks).map(|_| Chunk::empty()).collect(),
            topology,
            // v_chunks,
            // h_chunks,
        }
    }

    pub fn random(h_chunks: usize, v_chunks: usize, rng: &mut impl Rng) -> Self {
        Self::from_fn(h_chunks, v_chunks, |_, _| Chunk::random(rng))
    }

    pub fn empty(h_chunks: usize, v_chunks: usize) -> Self {
        Self::from_fn(h_chunks, v_chunks, |_, _| Chunk::empty())
    }

    pub fn num_cells(&self) -> usize {
        self.chunks.len() * chunk::SIZE * chunk::SIZE
    }

    pub fn num_alive(&self) -> usize {
        self.chunks.iter().map(Chunk::num_alive).sum()
    }

    pub fn clock(&mut self) {
        let src = self.chunks.as_ref();
        let dst = self.alternate.as_mut();
        let topo = self.topology.as_ref();

        let process =
            |((src_chunk, dst_chunk), links): ((&Chunk, &mut Chunk), &NeighborData<usize>)| {
                let nbhd = links.map(|idx| unsafe { src.get_unchecked(idx) });
                src_chunk.apply_rules(nbhd, dst_chunk);
            };

        if cfg!(feature = "rayon") && self.chunks.len() >= 16 {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;

                src.par_iter().zip(dst).zip(topo).for_each(process);
            }
        } else {
            src.iter().zip(dst).zip(topo).for_each(process);
        }

        mem::swap(&mut self.chunks, &mut self.alternate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    use crate::chunk::Dir;
    use crate::naive::chunk::Chunk as OldChunk;

    #[test]
    fn test_toroidal_world_matches_naive() {
        let (h, v) = (2, 2);
        let mut rng = StdRng::seed_from_u64(42);

        let mut world = ToroidalWorld::random(h, v, &mut rng);

        // Construct the matching naive world
        let mut old_chunks: Vec<OldChunk> = world
            .chunks
            .iter()
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
            for (idx, new_chunk) in world.chunks.iter().enumerate() {
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
