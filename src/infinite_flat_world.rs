use std::iter::zip;
use std::mem;

use fxhash::{FxHashMap, FxHashSet};

use crate::{
    Activity, Arena, Backend, Batch, Buffer, Coord, DefaultBackend, Dir, NeighborData, Orth, Owned,
};

type Links = NeighborData<Option<usize>>;

struct Node<B: Buffer> {
    coord: Coord,
    chunk: B::Owned,
    links: Links,
    activity: B::Activity,
}

struct RawInfiniteFlatWorld<B: Backend> {
    arena: Arena<B::Buffer>,
    nodes: Vec<Node<B::Buffer>>,
    index: FxHashMap<Coord, usize>,
    empty: <B::Buffer as Buffer>::Owned,
}

fn add(a: (i64, i64), b: (i64, i64)) -> (i64, i64) {
    (a.0 + b.0, a.1 + b.1)
}

impl<B: Backend> RawInfiniteFlatWorld<B> {
    #[allow(unused)]
    #[cfg(debug_assertions)]
    fn check_invariants(&self) {
        use crate::traits::Chunk;

        assert_eq!(self.index.len(), self.nodes.len());

        for (node_id, node) in self.nodes.iter().enumerate() {
            assert_eq!(node_id, self.index[&node.coord]);
            for (dir, &neighbor) in node.links.iter() {
                assert_eq!(neighbor, self.index.get(&node.coord.step(dir)).copied());
                if let Some(neighbor) = neighbor {
                    assert_eq!(Some(node_id), self.nodes[neighbor].links[dir.opposite()]);
                }
            }
        }

        assert!(unsafe { self.arena.load(self.empty.id()).is_empty() });
    }

    fn create(&mut self, coord: Coord, chunk: <B::Buffer as Buffer>::Owned, activity: B::Activity) {
        let node_id = self.nodes.len();

        let links = Links::from_fn(|dir| self.index.get(&coord.step(dir)).copied());
        self.nodes.push(Node {
            coord,
            chunk,
            links,
            activity,
        });
        self.index.insert(coord, node_id);

        // we also need to sync up the neighbors
        for (dir, &neighbor) in links.iter() {
            if let Some(neighbor) = neighbor {
                self.nodes[neighbor].links[dir.opposite()] = Some(node_id);
            }
        }
    }

    fn destroy(&mut self, node_id: usize) {
        let &Node { coord, links, .. } = &self.nodes[node_id];

        for (dir, &neighbor) in links.iter() {
            if let Some(neighbor) = neighbor {
                self.nodes[neighbor].links[dir.opposite()] = None;
            }
        }
        self.index.remove(&coord);
        self.arena.dealloc(self.nodes.swap_remove(node_id).chunk);

        // now we need to patch up the node that got replaced, only if it was not the last one
        let Some(&Node { coord, links, .. }) = self.nodes.get(node_id) else {
            return;
        };
        for (dir, &neighbor) in links.iter() {
            if let Some(neighbor) = neighbor {
                self.nodes[neighbor].links[dir.opposite()] = Some(node_id);
            }
        }
        self.index.insert(coord, node_id);
    }

    fn clock(&mut self, scratch: &mut Scratch<B::Buffer>) -> (usize, usize) {
        macro_rules! get_neighbors {
            ($links: expr) => {
                $links.map(|neighbor| {
                    neighbor
                        .map_or(&self.empty, |node_id| unsafe {
                            &self.nodes.get_unchecked(node_id).chunk
                        })
                        .id()
                })
            };
        }

        let mut batch = mem::take(&mut scratch.batch).recycle();

        scratch.checked.clear();

        let node_count = self.nodes.len();
        batch.reserve_alive(node_count);
        scratch.alive_dsts.reserve(node_count);

        for (node_id, node) in self.nodes.iter().enumerate() {
            batch.push_alive_within_capacity(node.chunk.id(), get_neighbors!(node.links));
            let _ = scratch.alive_dsts.push_within_capacity(self.arena.alloc());

            for orth in [Orth::N, Orth::W, Orth::E, Orth::S] {
                if !node.activity.edge(orth) {
                    continue;
                }

                let dir = orth.dir();

                if node.links[dir].is_some() {
                    continue;
                }

                let coord = node.coord.step(dir);
                if !scratch.checked.insert(coord) {
                    continue;
                }

                let links = Links::from_fn(|other_dir| {
                    let (dx, dy) = add(dir.offset(), other_dir.offset());
                    match (dx, dy) {
                        (0, 0) => Some(node_id),
                        (-1..=1, -1..=1) => node.links[Dir::from_offset(dx, dy)],
                        _ => self.index.get(&coord.step(other_dir)).copied(),
                    }
                });

                scratch.coords_buf.push(coord);
                batch.push_dead(get_neighbors!(links));
                scratch.dead_dsts.push(self.arena.alloc());
            }
        }

        unsafe {
            self.arena.compute_batched(
                &batch,
                &mut scratch.alive_dsts,
                &mut scratch.dead_dsts,
                &mut scratch.results_buf,
            )
        };

        let (alive_chunks_checked, dead_chunks_checked) =
            (scratch.alive_dsts.len(), scratch.dead_dsts.len());

        scratch.batch = batch.recycle();

        for (node_id, (chunk, activity)) in zip(
            scratch.alive_dsts.drain(..),
            scratch.results_buf.drain(..self.nodes.len()),
        )
        .enumerate()
        .rev()
        {
            if activity.chunk() {
                let node = unsafe { self.nodes.get_unchecked_mut(node_id) };
                self.arena.dealloc(mem::replace(&mut node.chunk, chunk));
                node.activity = activity;
            } else {
                self.arena.dealloc(chunk);
                self.destroy(node_id);
            }
        }

        for (coord, (chunk, activity)) in zip(
            scratch.coords_buf.drain(..),
            zip(scratch.dead_dsts.drain(..), scratch.results_buf.drain(..)),
        ) {
            if activity.chunk() {
                self.create(coord, chunk, activity);
            } else {
                self.arena.dealloc(chunk);
            }
        }

        #[cfg(debug_assertions)]
        self.check_invariants();

        (alive_chunks_checked, dead_chunks_checked)
    }
}

struct Scratch<B: Buffer> {
    checked: FxHashSet<Coord>,
    coords_buf: Vec<Coord>,
    batch: Batch<'static, B>,
    alive_dsts: Vec<B::Owned>,
    dead_dsts: Vec<B::Owned>,
    results_buf: Vec<B::Activity>,
}

impl<B: Buffer> Default for Scratch<B> {
    fn default() -> Self {
        Self {
            checked: Default::default(),
            coords_buf: Default::default(),
            batch: Default::default(),
            alive_dsts: Default::default(),
            dead_dsts: Default::default(),
            results_buf: Default::default(),
        }
    }
}

pub struct InfiniteFlatWorld<B: Backend = DefaultBackend> {
    raw: RawInfiniteFlatWorld<B>,
    scratch: Scratch<B::Buffer>,
}

impl<B: Backend> InfiniteFlatWorld<B> {
    pub fn empty(config: <B::Buffer as Buffer>::Config) -> Self {
        let mut arena = Arena::<B::Buffer>::new(config);
        Self {
            raw: RawInfiniteFlatWorld {
                empty: arena.alloc_zeroed(),
                arena,
                nodes: Vec::new(),
                index: FxHashMap::default(),
            },
            scratch: Scratch::<B::Buffer>::default(),
        }
    }

    pub fn insert_chunk(&mut self, coord: Coord, chunk: B::Chunk) {
        if let Some(&node_id) = self.raw.index.get(&coord) {
            let node = &mut self.raw.nodes[node_id];
            unsafe { self.raw.arena.store(&mut node.chunk, &chunk) };
            node.activity = Activity::ALL;
        } else {
            let chunk = self.raw.arena.own(chunk);
            self.raw.create(coord, chunk, Activity::ALL);
        }
    }

    pub fn load_chunk(&self, coord: Coord) -> Option<B::Chunk> {
        Some(unsafe {
            self.raw
                .arena
                .load(self.raw.nodes[*self.raw.index.get(&coord)?].chunk.id())
        })
    }

    pub fn load_chunk_to<'a>(
        &self,
        coord: Coord,
        dst: &'a mut B::Chunk,
    ) -> Option<&'a mut B::Chunk> {
        unsafe {
            self.raw
                .arena
                .load_to(self.raw.nodes[*self.raw.index.get(&coord)?].chunk.id(), dst)
        };
        Some(dst)
    }

    pub fn active_chunks(&self) -> usize {
        self.raw.nodes.len()
    }

    pub fn clock(&mut self) -> (usize, usize) {
        self.raw.clock(&mut self.scratch)
    }
}

impl<B: Backend> Clone for InfiniteFlatWorld<B> {
    fn clone(&self) -> Self {
        let mut new_arena = self.raw.arena.empty_clone();

        let new_nodes = self
            .raw
            .nodes
            .iter()
            .map(|node| {
                let mut new_chunk = new_arena.alloc();
                unsafe { new_arena.copy(&mut new_chunk, &self.raw.arena, node.chunk.id()) };

                Node {
                    coord: node.coord,
                    chunk: new_chunk,
                    links: node.links,
                    activity: node.activity,
                }
            })
            .collect();

        Self {
            raw: RawInfiniteFlatWorld {
                empty: new_arena.alloc_zeroed(),
                arena: new_arena,
                nodes: new_nodes,
                index: self.raw.index.clone(),
            },
            scratch: Scratch::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rand::{SeedableRng, rngs::StdRng};
    use std::collections::HashSet;

    // Adjust imports as necessary for your module structure
    use crate::backend::{SimdBackend, SimdChunk as NewChunk};
    use crate::infinite_flat_world::{Coord, InfiniteFlatWorld as NewWorld};

    use crate::naive::chunk::NaiveChunk as OldChunk;
    use crate::naive::infinite_flat_world::InfiniteFlatWorld as OldWorld;
    use crate::traits::Chunk;

    #[test]
    fn test_infinite_flat_world_topology_matches_naive() {
        let mut rng = StdRng::seed_from_u64(1337);
        let mut old_world = OldWorld::empty();
        let mut new_world = NewWorld::<SimdBackend>::empty(());

        // 1. Seed a dense cluster of random chunks around the origin
        let seed_coords = [(0, 0), (1, 0), (0, 1), (-1, -1), (1, 1)];
        for &(cx, cy) in &seed_coords {
            let new_chunk = NewChunk::from_bytes(&rng.random());
            let old_chunk = OldChunk::from_bytes(new_chunk.to_bytes());

            new_world.insert_chunk(Coord { x: cx, y: cy }, new_chunk);
            old_world
                .active_chunks
                .insert((cx as i128, cy as i128), old_chunk);
        }

        // 2. Tick both simulations forward heavily to force border expansions and chunk culling
        for i in 1..=15 {
            old_world.clock();
            new_world.clock();

            // 3. Verify that the optimized world successfully routed and tracked every single active chunk
            for (&(cx, cy), old_chunk) in &old_world.active_chunks {
                let new_chunk = new_world
                    .load_chunk(Coord {
                        x: cx as i64,
                        y: cy as i64,
                    })
                    .unwrap_or_else(|| {
                        panic!("Gen {i}: Optimized world missing chunk at ({cx}, {cy})")
                    });

                let old_from_new = OldChunk::from_bytes(new_chunk.to_bytes());

                assert_eq!(
                    old_chunk, &old_from_new,
                    "Gen {i}: Mismatch inside chunk ({cx}, {cy})"
                );
            }

            // 4. Verify the optimized world didn't leak "ghost" chunks that should have been pruned
            // Extract the active coordinate footprint from both engines
            let new_chunk_coords: HashSet<(i128, i128)> = new_world
                .raw
                .nodes
                .iter()
                .map(|n| (n.coord.x as i128, n.coord.y as i128))
                .collect();

            let old_chunk_coords: HashSet<(i128, i128)> =
                old_world.active_chunks.keys().copied().collect();

            assert_eq!(
                new_chunk_coords, old_chunk_coords,
                "Gen {i}: Active chunk footprint mismatch! The Arena routing diverged from the naive hashmap."
            );
        }
    }
}
