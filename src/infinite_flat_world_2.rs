use std::iter::zip;
use std::mem;

use fxhash::{FxHashMap, FxHashSet};

use crate::arena::{Arena, Batch, ChunkRef};
use crate::chunk::{Activity, Chunk, Dir, NeighborData};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord {
    pub x: i64,
    pub y: i64,
}

impl Coord {
    pub fn step(self, dir: Dir) -> Self {
        let (dx, dy) = dir.offset();
        Self {
            x: self.x + dx,
            y: self.y + dy,
        }
    }
}

type Links = NeighborData<Option<usize>>;

struct Node {
    coord: Coord,
    chunk: ChunkRef,
    links: Links,
    activity: Activity,
}

struct RawInfiniteFlatWorld {
    arena: Arena,

    nodes: Vec<Node>,
    index: FxHashMap<Coord, usize>,

    empty: ChunkRef,
}

fn add(a: (i64, i64), b: (i64, i64)) -> (i64, i64) {
    (a.0 + b.0, a.1 + b.1)
}

impl RawInfiniteFlatWorld {
    #[allow(unused)]
    #[cfg(debug_assertions)]
    fn check_invariants(&self) {
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
    }

    fn create(&mut self, coord: Coord, chunk: ChunkRef, activity: Activity) {
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

    fn clock(
        &mut self,
        checked: &mut FxHashSet<Coord>,
        coords_buf: &mut Vec<Coord>,
        static_batch: &mut Batch<'static>,
        alive_dsts: &mut Vec<ChunkRef>,
        dead_dsts: &mut Vec<ChunkRef>,
        results_buf: &mut Vec<Activity>,
    ) {
        macro_rules! get_neighbors {
            ($links: expr) => {
                $links.map(|neighbor| {
                    neighbor
                        .map_or(&self.empty, |node_id| &self.nodes[node_id].chunk)
                        .read()
                })
            };
        }

        let mut batch = mem::take(static_batch).reinterpret();

        checked.clear();

        for (node_id, node) in self.nodes.iter().enumerate() {
            batch.push_alive(node.chunk.read(), get_neighbors!(node.links));
            alive_dsts.push(self.arena.alloc());

            for (dir, &neighbor) in node.links.iter() {
                if neighbor.is_some() {
                    continue;
                }

                let coord = node.coord.step(dir);
                if checked.contains(&coord) {
                    continue;
                }

                checked.insert(coord);

                let links = Links::from_fn(|other_dir| {
                    let (dx, dy) = add(dir.offset(), other_dir.offset());
                    match (dx, dy) {
                        (0, 0) => Some(node_id),
                        (-1..=1, -1..=1) => self.nodes[node_id].links[Dir::from_offset(dx, dy)],
                        _ => self.index.get(&coord.step(other_dir)).copied(),
                    }
                });

                if ![Dir::N, Dir::W, Dir::E, Dir::S].into_iter().any(|dir| {
                    links[dir].map_or(false, |node_id| {
                        self.nodes[node_id].activity.is_edge_active(dir.opposite())
                    })
                }) {
                    // if no adjacent edges are active then we don't need to check anything
                    continue;
                }

                coords_buf.push(coord);
                batch.push_dead(get_neighbors!(links));
                dead_dsts.push(self.arena.alloc());
            }
        }

        self.arena
            .compute_batched(&batch, alive_dsts, dead_dsts, results_buf);

        *static_batch = batch.reinterpret();

        for (node_id, (chunk, activity)) in
            zip(alive_dsts.drain(..), results_buf.drain(..self.nodes.len()))
                .enumerate()
                .rev()
        {
            if activity.is_active() {
                let node = &mut self.nodes[node_id];
                self.arena.dealloc(mem::replace(&mut node.chunk, chunk));
                node.activity = activity;
            } else {
                self.arena.dealloc(chunk);
                self.destroy(node_id);
            }
        }

        for (coord, (chunk, activity)) in zip(
            coords_buf.drain(..),
            zip(dead_dsts.drain(..), results_buf.drain(..)),
        ) {
            if activity.is_active() {
                self.create(coord, chunk, activity);
            } else {
                self.arena.dealloc(chunk);
            }
        }
    }
}

pub struct InfiniteFlatWorld {
    raw: RawInfiniteFlatWorld,
    checked: FxHashSet<Coord>,
    coords_buf: Vec<Coord>,
    batch: Batch<'static>,
    alive_dsts: Vec<ChunkRef>,
    dead_dsts: Vec<ChunkRef>,
    results_buf: Vec<Activity>,
}

impl InfiniteFlatWorld {
    pub fn empty() -> Self {
        let mut arena = Arena::new();
        Self {
            raw: RawInfiniteFlatWorld {
                empty: arena.alloc_zeroed(),
                arena,
                nodes: Vec::new(),
                index: FxHashMap::default(),
            },
            checked: FxHashSet::default(),
            coords_buf: Vec::new(),
            batch: Batch::new(),
            alive_dsts: Vec::new(),
            dead_dsts: Vec::new(),
            results_buf: Vec::new(),
        }
    }

    pub fn insert_chunk(&mut self, coord: Coord, chunk: Chunk) {
        let chunk = self.raw.arena.own(chunk);

        match self.raw.index.get(&coord) {
            Some(&node_id) => {
                let node = &mut self.raw.nodes[node_id];
                node.chunk = chunk;
                node.activity = Activity::ALL;
            }
            None => self.raw.create(coord, chunk, Activity::ALL),
        }
    }

    pub fn load_chunk(&self, coord: Coord) -> Option<Chunk> {
        Some(
            self.raw
                .arena
                .load(self.raw.nodes[*self.raw.index.get(&coord)?].chunk.read()),
        )
    }

    pub fn load_chunk_to<'a>(&self, coord: Coord, dst: &'a mut Chunk) -> Option<&'a mut Chunk> {
        self.raw.arena.load_to(
            self.raw.nodes[*self.raw.index.get(&coord)?].chunk.read(),
            dst,
        );
        Some(dst)
    }

    pub fn clock(&mut self) {
        self.raw.clock(
            &mut self.checked,
            &mut self.coords_buf,
            &mut self.batch,
            &mut self.alive_dsts,
            &mut self.dead_dsts,
            &mut self.results_buf,
        );
    }
}

impl Clone for InfiniteFlatWorld {
    fn clone(&self) -> Self {
        let mut new_arena = Arena::new();

        let new_nodes = self
            .raw
            .nodes
            .iter()
            .map(|node| {
                let mut new_chunk = new_arena.alloc();
                new_arena.copy(&mut new_chunk, &self.raw.arena, node.chunk.read());

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
            checked: self.checked.clone(),
            coords_buf: Vec::new(),
            batch: Batch::new(),
            alive_dsts: Vec::new(),
            dead_dsts: Vec::new(),
            results_buf: Vec::new(),
        }
    }
}
