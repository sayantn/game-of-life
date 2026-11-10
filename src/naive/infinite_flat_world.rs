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
        let empty = Chunk::empty();

        macro_rules! get {
            ($x:expr, $y:expr) => {
                self.active_chunks.get(&($x, $y)).copied()
            };
        }
        macro_rules! or_empty {
            ($target:ident . $name:ident ($($chunk:expr),* $(,)?)) => {
                $target.$name($(&$chunk.unwrap_or(empty)),*)
            };
            ($target:ident :: $name:ident ($($chunk:expr),* $(,)?)) => {
                $target::$name($(&$chunk.unwrap_or(empty)),*)
            }
        }

        let active_chunk_coords: Vec<_> = self.active_chunks.keys().copied().collect();

        for &(x, y) in &active_chunk_coords {
            let top = get!(x, y + 1);
            let top_right = get!(x + 1, y + 1);
            let right = get!(x + 1, y);
            let bottom_right = get!(x + 1, y - 1);
            let bottom = get!(x, y - 1);
            let bottom_left = get!(x - 1, y - 1);
            let left = get!(x - 1, y);
            let top_left = get!(x - 1, y + 1);

            let chunk = self.active_chunks.get_mut(&(x, y)).unwrap();

            or_empty!(chunk.check(
                top,
                top_right,
                right,
                bottom_right,
                bottom,
                bottom_left,
                left,
                top_left,
            ));

            let chunk = Some(*chunk);

            if top.is_none() {
                if let Some(new_chunk) = or_empty!(Chunk::check_borders(
                    get!(x, y + 2),
                    get!(x + 1, y + 2),
                    top_right,
                    right,
                    chunk,
                    left,
                    top_left,
                    get!(x - 1, y + 2),
                )) {
                    self.active_chunks.insert((x, y + 1), new_chunk);
                }
            }
            if top_right.is_none() {
                if let Some(new_chunk) = or_empty!(Chunk::check_borders(
                    get!(x + 1, y + 2),
                    get!(x + 2, y + 2),
                    get!(x + 2, y + 1),
                    get!(x + 2, y),
                    right,
                    chunk,
                    top,
                    get!(x, y + 2),
                )) {
                    self.active_chunks.insert((x + 1, y + 1), new_chunk);
                }
            }
            if right.is_none() {
                if let Some(new_chunk) = or_empty!(Chunk::check_borders(
                    top_right,
                    get!(x + 2, y + 1),
                    get!(x + 2, y),
                    get!(x + 2, y - 1),
                    bottom_right,
                    bottom,
                    chunk,
                    top
                )) {
                    self.active_chunks.insert((x + 1, y), new_chunk);
                }
            }
            if bottom_right.is_none() {
                if let Some(new_chunk) = or_empty!(Chunk::check_borders(
                    right,
                    get!(x + 2, y),
                    get!(x + 2, y - 1),
                    get!(x + 2, y - 2),
                    get!(x + 1, y - 2),
                    get!(x, y - 2),
                    bottom,
                    chunk
                )) {
                    self.active_chunks.insert((x + 1, y - 1), new_chunk);
                }
            }
            if bottom.is_none() {
                if let Some(new_chunk) = or_empty!(Chunk::check_borders(
                    chunk,
                    right,
                    bottom_right,
                    get!(x + 1, y - 2),
                    get!(x, y - 2),
                    get!(x - 1, y - 2),
                    bottom_left,
                    left
                )) {
                    self.active_chunks.insert((x, y - 1), new_chunk);
                }
            }
            if bottom_left.is_none() {
                if let Some(new_chunk) = or_empty!(Chunk::check_borders(
                    left,
                    chunk,
                    bottom,
                    get!(x, y - 2),
                    get!(x - 1, y - 2),
                    get!(x - 2, y - 2),
                    get!(x - 2, y - 1),
                    get!(x - 2, y),
                )) {
                    self.active_chunks.insert((x - 1, y - 1), new_chunk);
                }
            }
            if left.is_none() {
                if let Some(new_chunk) = or_empty!(Chunk::check_borders(
                    top_left,
                    top,
                    chunk,
                    bottom,
                    bottom_left,
                    get!(x - 2, y - 1),
                    get!(x - 2, y),
                    get!(x - 2, y + 1)
                )) {
                    self.active_chunks.insert((x - 1, y), new_chunk);
                }
            }
            if top_left.is_none() {
                if let Some(new_chunk) = or_empty!(Chunk::check_borders(
                    get!(x - 1, y + 2),
                    get!(x, y + 2),
                    top,
                    chunk,
                    left,
                    get!(x - 2, y),
                    get!(x - 2, y + 1),
                    get!(x - 2, y + 2)
                )) {
                    self.active_chunks.insert((x - 1, y + 1), new_chunk);
                }
            }
        }

        self.active_chunks.retain(|_, chunk| chunk.will_be_active());

        for chunk in self.active_chunks.values_mut() {
            chunk.update();
        }
    }
}
