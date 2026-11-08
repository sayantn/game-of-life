use std::mem;
use std::ops::Index;

use fxhash::{FxHashMap, FxHashSet};

use crate::arena::{Arena, ChunkRef};
use crate::chunk::{Chunk, Dir, NeighborData};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord {
    pub x: i64,
    pub y: i64,
}

impl Coord {
    fn step(self, dir: Dir) -> Self {
        let (dx, dy) = dir.offset();
        Self {
            x: self.x + dx,
            y: self.y + dy,
        }
    }
}

type Links = NeighborData<Option<usize>>;

#[derive(Clone, Copy)]
struct Meta {
    coord: Coord,
    links: Links,
    edge_activity: [bool; 2],
}

pub struct InfiniteFlatWorld {
    arena: Arena,

    current: Vec<ChunkRef>,
    next: Vec<ChunkRef>,

    meta: Vec<Meta>,

    index: FxHashMap<Coord, usize>,
    empty: Box<Chunk>,

    dying: Vec<usize>,
    still_dead: FxHashSet<Coord>,
    newborn: FxHashMap<Coord, (ChunkRef, [bool; 2])>,
}

fn add(a: (i64, i64), b: (i64, i64)) -> (i64, i64) {
    (a.0 + b.0, a.1 + b.1)
}

impl InfiniteFlatWorld {
    pub fn empty() -> Self {
        Self {
            arena: Arena::new(),
            current: Vec::new(),
            next: Vec::new(),
            meta: Vec::new(),
            index: FxHashMap::default(),
            empty: Box::new(Chunk::empty()),
            dying: Vec::new(),
            still_dead: FxHashSet::default(),
            newborn: FxHashMap::default(),
        }
    }

    pub fn insert_chunk(&mut self, coord: Coord, chunk: Chunk) {
        let chunk = self.arena.own(chunk);
        self.newborn.insert(coord, (chunk, [true; 2])); // be pessimistic about added chunks
        self.create();
    }

    #[allow(unused)]
    #[cfg(debug_assertions)]
    fn check_invariants(&self) {
        assert_eq!(self.current.len(), self.meta.len());
        assert_eq!(self.next.len(), self.meta.len());
        assert_eq!(self.index.len(), self.meta.len());

        for (node_id, meta) in self.meta.iter().enumerate() {
            assert_eq!(node_id, self.index[&meta.coord]);
            for (dir, &neighbor) in meta.links.iter() {
                assert_eq!(neighbor, self.index.get(&meta.coord.step(dir)).copied());
                if let Some(neighbor) = neighbor {
                    assert_eq!(Some(node_id), self.meta[neighbor].links[dir.opposite()]);
                }
            }
        }

        assert!(self.empty.is_empty());
    }

    fn create(&mut self) {
        for (coord, (chunk_ref, edge_activity)) in self.newborn.drain() {
            let node_id = self.current.len();

            // first just insert it, insert any random junk into `next` to keep the length synced
            self.current.push(chunk_ref);
            self.next.push(self.arena.alloc());
            self.index.insert(coord, node_id);

            // then we need to sync up the neighbors table
            let links = Links::from_fn(|dir| self.index.get(&coord.step(dir)).copied());
            self.meta.push(Meta {
                coord,
                links,
                edge_activity,
            });
            for (dir, &neighbor) in links.iter() {
                if let Some(neighbor) = neighbor {
                    self.meta[neighbor].links[dir.opposite()] = Some(node_id);
                }
            }
        }
    }

    fn destroy(&mut self) {
        for node_id in self.dying.drain(..).rev() {
            // prepare for removal, patch up the links and coord-nodeid map
            let meta = self.meta[node_id];
            for (dir, &neighbor) in meta.links.iter() {
                if let Some(neighbor) = neighbor {
                    self.meta[neighbor].links[dir.opposite()] = None;
                }
            }
            self.index.remove(&meta.coord);

            // then we swap-remove it
            self.arena.dealloc(self.current.swap_remove(node_id));
            self.arena.dealloc(self.next.swap_remove(node_id));
            self.meta.swap_remove(node_id);

            // now we need to patch up the node that got replaced, only if it was not the last one
            let Some(&meta) = self.meta.get(node_id) else {
                continue;
            };
            for (dir, &neighbor) in meta.links.iter() {
                if let Some(neighbor) = neighbor {
                    self.meta[neighbor].links[dir.opposite()] = Some(node_id);
                }
            }
            self.index.insert(meta.coord, node_id);
        }
    }

    pub fn clock(&mut self) {
        let mut scratch_ref = self.arena.alloc();
        let mut scratch = self.arena.get_mut(&mut scratch_ref);

        macro_rules! get_neighbors {
            ($links: expr) => {
                $links.map(|neighbor| {
                    neighbor.map_or(&*self.empty, |node_id| {
                        self.arena.get(&self.current[node_id])
                    })
                })
            };
        }

        for (node_id, &meta) in self.meta.iter().enumerate() {
            for (dir, &neighbor) in meta.links.iter() {
                if neighbor.is_some() {
                    continue;
                }

                let coord = meta.coord.step(dir);
                if self.still_dead.contains(&coord) || self.newborn.contains_key(&coord) {
                    continue;
                }

                let links = Links::from_fn(|other_dir| {
                    let (dx, dy) = add(dir.offset(), other_dir.offset());
                    match (dx, dy) {
                        (0, 0) => Some(node_id),
                        (-1..=1, -1..=1) => self.meta[node_id].links[Dir::from_offset(dx, dy)],
                        _ => self.index.get(&coord.step(other_dir)).copied(),
                    }
                });

                let neighbors = get_neighbors!(links);

                if ![Dir::N, Dir::W, Dir::E, Dir::S]
                    .into_iter()
                    .filter_map(|dir| links[dir].map(|node_id| (dir, node_id)))
                    .any(|(dir, node_id)| match dir {
                        Dir::N => neighbors[Dir::N].is_bottom_active(),
                        Dir::W => self.meta[node_id].edge_activity[1],
                        Dir::E => self.meta[node_id].edge_activity[0],
                        Dir::S => neighbors[Dir::S].is_top_active(),
                        _ => unreachable!(),
                    })
                {
                    // if no adjacent edges are active then we don't need to check anything
                    self.still_dead.insert(coord);
                    continue;
                }

                let [left_edge_active, chunk_alive, right_edge_active] =
                    Chunk::check_activity(neighbors, scratch);

                if chunk_alive {
                    self.newborn
                        .insert(coord, (scratch_ref, [left_edge_active, right_edge_active]));

                    scratch_ref = self.arena.alloc();
                    scratch = self.arena.get_mut(&mut scratch_ref);
                } else {
                    self.still_dead.insert(coord);
                }
            }
        }

        for (node_id, meta) in self.meta.iter_mut().enumerate() {
            let nbhd = get_neighbors!(meta.links);
            let dst = self.arena.get_mut(&mut self.next[node_id]);

            let [left_edge_active, chunk_alive, right_edge_active] = self
                .arena
                .get(&self.current[node_id])
                .apply_rules(nbhd, dst);

            if chunk_alive {
                meta.edge_activity = [left_edge_active, right_edge_active];
            } else {
                self.dying.push(node_id);
            }
        }

        self.arena.dealloc(scratch_ref);
        self.still_dead.clear();

        mem::swap(&mut self.current, &mut self.next);

        self.destroy();
        self.create();
    }
}

impl Index<Coord> for InfiniteFlatWorld {
    type Output = Chunk;

    fn index(&self, index: Coord) -> &Self::Output {
        self.arena.get(&self.current[self.index[&index]])
    }
}
