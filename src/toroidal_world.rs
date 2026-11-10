use std::mem;

use rand::Rng;
use rayon::prelude::*;

use crate::chunk::{self, Chunk, NeighborData};

#[derive(Clone)]
pub struct ToroidalWorld {
    chunks: Box<[Chunk]>,
    alternate: Box<[Chunk]>,
    v_chunks: usize,
    h_chunks: usize,
}

impl ToroidalWorld {
    pub fn from_fn(
        h_chunks: usize,
        v_chunks: usize,
        mut f: impl FnMut(usize, usize) -> Chunk,
    ) -> Self {
        Self {
            chunks: (0..v_chunks)
                .flat_map(|i| (0..h_chunks).map(move |j| (i, j)))
                .map(|(i, j)| f(i, j))
                .collect(),
            alternate: (0..v_chunks * h_chunks).map(|_| Chunk::empty()).collect(),
            v_chunks,
            h_chunks,
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
        let (v, h) = (self.v_chunks, self.h_chunks);
        let src = self.chunks.as_ref();
        let dst = self.alternate.as_mut();

        let process = |(idx, dst_chunk): (usize, &mut Chunk)| {
            // Reconstruct 2D coordinates from the 1D index
            let i = idx / h;
            let j = idx % h;

            let rows = [((i + v - 1) % v) * h, i * h, ((i + 1) % v) * h];
            let cols = [(j + h - 1) % h, j, (j + 1) % h];

            let nbhd = NeighborData::from_fn(|d| {
                let (dx, dy) = d.offset();
                &src[rows[(dy + 1) as usize] + cols[(dx + 1) as usize]]
            });

            src[idx].apply_rules(nbhd, dst_chunk);
        };

        if src.len() <= 16 {
            dst.iter_mut().enumerate().for_each(process);
        } else {
            dst.par_iter_mut().enumerate().for_each(process);
        }

        mem::swap(&mut self.chunks, &mut self.alternate);
    }
}

#[cfg(test)]
mod oracle {
    use super::*;
    use crate::chunk::SIZE;
    use rand::{SeedableRng, rngs::StdRng};

    /// Flatten the world into a global bool grid, row-major.
    pub fn cells(w: &ToroidalWorld) -> Vec<Vec<bool>> {
        let (vh, hh) = (w.v_chunks, w.h_chunks);
        let mut g = vec![vec![false; hh * SIZE]; vh * SIZE];
        for ci in 0..vh {
            for cj in 0..hh {
                let bytes = w.chunks[ci * hh + cj].to_bytes();
                for r in 0..SIZE {
                    for c in 0..SIZE {
                        g[ci * SIZE + r][cj * SIZE + c] = bytes[r][c / 8] & (0x80 >> (c % 8)) != 0;
                    }
                }
            }
        }
        g
    }

    /// One generation of GoL on a torus, written as plainly as possible.
    pub fn step(g: &[Vec<bool>]) -> Vec<Vec<bool>> {
        let (h, w) = (g.len(), g[0].len());
        let mut out = vec![vec![false; w]; h];
        for y in 0..h {
            for x in 0..w {
                let mut n = 0;
                for dy in 0..3 {
                    for dx in 0..3 {
                        if dy == 1 && dx == 1 {
                            continue;
                        }
                        let yy = (y + h + dy - 1) % h;
                        let xx = (x + w + dx - 1) % w;
                        n += g[yy][xx] as u8;
                    }
                }
                out[y][x] = matches!((g[y][x], n), (true, 2) | (_, 3));
            }
        }
        out
    }

    #[test]
    fn matches_naive_on_random_worlds() {
        for seed in 0..8u64 {
            let mut world = ToroidalWorld::random(2, 2, &mut StdRng::seed_from_u64(seed));
            let mut expected = cells(&world);

            for i in 0..50 {
                world.clock();
                expected = step(&expected);
                assert_eq!(cells(&world), expected, "seed {seed}, generation {i}");
            }
        }
    }
}
