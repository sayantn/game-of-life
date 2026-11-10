use super::cell::Cell::{self, *};
use std::{collections::HashMap, ops::Index};

use super::chunk::{Chunk, MASK_X, MASK_Y, SHIFT_X, SHIFT_Y};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct InfiniteFlatWorld {
    pub active_chunks: HashMap<(i128, i128), Chunk>,
}

impl Index<(i128, i128)> for InfiniteFlatWorld {
    type Output = Cell;

    fn index(&self, (x, y): (i128, i128)) -> &Self::Output {
        let chunk_coords = (x >> SHIFT_X, y >> SHIFT_Y);
        self.active_chunks
            .get(&chunk_coords)
            .map(|chunk| &chunk[(x as usize & MASK_X, y as usize & MASK_Y)])
            .unwrap_or(&Dead)
    }
}

impl InfiniteFlatWorld {
    pub fn empty() -> Self {
        Self {
            active_chunks: HashMap::new(),
        }
    }

    pub fn toggle(&mut self, x: i128, y: i128) {
        self.active_chunks
            .entry((x >> SHIFT_X, y >> SHIFT_Y))
            .or_insert_with(Chunk::empty)
            .toggle(x as usize & MASK_X, y as usize & MASK_Y);
    }

    pub fn clock(&mut self) {
        // We only allocate one empty chunk per tick, keeping the stack lean
        let empty = Chunk::empty();
        let active_chunk_coords: Vec<_> = self.active_chunks.keys().copied().collect();

        for &(x, y) in &active_chunk_coords {
            // 1. Temporarily pop the center chunk out of the map.
            // This gives us an owned `mut` chunk without borrowing the map.
            let mut chunk = self.active_chunks.remove(&(x, y)).unwrap();

            // 2. A macro to inline map lookups and fall back to references of `empty`
            macro_rules! get {
                ($x:expr, $y:expr) => {
                    self.active_chunks.get(&($x, $y)).unwrap_or(&empty)
                };
            }

            // 3. Process the main rules. The immutable borrows of the map
            // end immediately when this function call finishes.
            chunk.check(
                get!(x, y + 1),
                get!(x + 1, y + 1),
                get!(x + 1, y),
                get!(x + 1, y - 1),
                get!(x, y - 1),
                get!(x - 1, y - 1),
                get!(x - 1, y),
                get!(x - 1, y + 1),
            );

            // 4. Process borders. Because we don't save `get!()` to a variable,
            // we are free to call `.insert()` immediately after!
            if !self.active_chunks.contains_key(&(x, y + 1)) {
                if let Some(new_chunk) = Chunk::check_borders(
                    get!(x, y + 2),
                    get!(x + 1, y + 2),
                    get!(x + 1, y + 1),
                    get!(x + 1, y),
                    &chunk,
                    get!(x - 1, y),
                    get!(x - 1, y + 1),
                    get!(x - 1, y + 2),
                ) {
                    self.active_chunks.insert((x, y + 1), new_chunk);
                }
            }
            if !self.active_chunks.contains_key(&(x + 1, y + 1)) {
                if let Some(new_chunk) = Chunk::check_borders(
                    get!(x + 1, y + 2),
                    get!(x + 2, y + 2),
                    get!(x + 2, y + 1),
                    get!(x + 2, y),
                    get!(x + 1, y),
                    &chunk,
                    get!(x, y + 1),
                    get!(x, y + 2),
                ) {
                    self.active_chunks.insert((x + 1, y + 1), new_chunk);
                }
            }
            if !self.active_chunks.contains_key(&(x + 1, y)) {
                if let Some(new_chunk) = Chunk::check_borders(
                    get!(x + 1, y + 1),
                    get!(x + 2, y + 1),
                    get!(x + 2, y),
                    get!(x + 2, y - 1),
                    get!(x + 1, y - 1),
                    get!(x, y - 1),
                    &chunk,
                    get!(x, y + 1),
                ) {
                    self.active_chunks.insert((x + 1, y), new_chunk);
                }
            }
            if !self.active_chunks.contains_key(&(x + 1, y - 1)) {
                if let Some(new_chunk) = Chunk::check_borders(
                    get!(x + 1, y),
                    get!(x + 2, y),
                    get!(x + 2, y - 1),
                    get!(x + 2, y - 2),
                    get!(x + 1, y - 2),
                    get!(x, y - 2),
                    get!(x, y - 1),
                    &chunk,
                ) {
                    self.active_chunks.insert((x + 1, y - 1), new_chunk);
                }
            }
            if !self.active_chunks.contains_key(&(x, y - 1)) {
                if let Some(new_chunk) = Chunk::check_borders(
                    &chunk,
                    get!(x + 1, y),
                    get!(x + 1, y - 1),
                    get!(x + 1, y - 2),
                    get!(x, y - 2),
                    get!(x - 1, y - 2),
                    get!(x - 1, y - 1),
                    get!(x - 1, y),
                ) {
                    self.active_chunks.insert((x, y - 1), new_chunk);
                }
            }
            if !self.active_chunks.contains_key(&(x - 1, y - 1)) {
                if let Some(new_chunk) = Chunk::check_borders(
                    get!(x - 1, y),
                    &chunk,
                    get!(x, y - 1),
                    get!(x, y - 2),
                    get!(x - 1, y - 2),
                    get!(x - 2, y - 2),
                    get!(x - 2, y - 1),
                    get!(x - 2, y),
                ) {
                    self.active_chunks.insert((x - 1, y - 1), new_chunk);
                }
            }
            if !self.active_chunks.contains_key(&(x - 1, y)) {
                if let Some(new_chunk) = Chunk::check_borders(
                    get!(x - 1, y + 1),
                    get!(x, y + 1),
                    &chunk,
                    get!(x, y - 1),
                    get!(x - 1, y - 1),
                    get!(x - 2, y - 1),
                    get!(x - 2, y),
                    get!(x - 2, y + 1),
                ) {
                    self.active_chunks.insert((x - 1, y), new_chunk);
                }
            }
            if !self.active_chunks.contains_key(&(x - 1, y + 1)) {
                if let Some(new_chunk) = Chunk::check_borders(
                    get!(x - 1, y + 2),
                    get!(x, y + 2),
                    get!(x, y + 1),
                    &chunk,
                    get!(x - 1, y),
                    get!(x - 2, y),
                    get!(x - 2, y + 1),
                    get!(x - 2, y + 2),
                ) {
                    self.active_chunks.insert((x - 1, y + 1), new_chunk);
                }
            }

            // 5. Put the mutated chunk back into the map safely!
            self.active_chunks.insert((x, y), chunk);
        }

        self.active_chunks.retain(|_, chunk| chunk.will_be_active());

        for chunk in self.active_chunks.values_mut() {
            chunk.update();
        }
    }
}
