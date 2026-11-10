use std::fmt::{Debug, Display, Write};
use std::iter::zip;
use std::mem;
use std::ops::Index;

use rand::Rng;

use crate::chunk::SIZE;

use super::cell::Cell::{self, *};

pub(super) const SHIFT_X: usize = SIZE.trailing_zeros() as _;
pub(super) const SHIFT_Y: usize = SHIFT_X;

pub(super) const SIZE_X: usize = 1 << SHIFT_X;
pub(super) const SIZE_Y: usize = 1 << SHIFT_Y;

pub(super) const MASK_X: usize = SIZE_X - 1;
pub(super) const MASK_Y: usize = SIZE_Y - 1;

#[derive(Copy, Clone, Eq)]
pub struct Chunk {
    data: [[Cell; SIZE_X]; SIZE_Y],
    alternate: [[Cell; SIZE_X]; SIZE_Y],
}

impl PartialEq for Chunk {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T> Index<(usize, T)> for Chunk
where
    [Cell]: Index<T>,
{
    type Output = <[Cell] as Index<T>>::Output;

    fn index(&self, (x, y): (usize, T)) -> &Self::Output {
        &self.data[x][y]
    }
}

fn is_active<const N: usize>(cells: &[[Cell; N]]) -> bool {
    cells
        .iter()
        .any(|row| row.iter().any(|alive| *alive == Alive))
}

impl Chunk {
    pub fn empty() -> Self {
        Self {
            data: [[Dead; _]; _],
            alternate: [[Dead; _]; _],
        }
    }

    pub fn random() -> Self {
        let mut chunk = Chunk::empty();
        let mut rng = rand::rng();
        for row in &mut chunk.data {
            for cell in row {
                *cell = rng.random::<bool>().into();
            }
        }
        chunk
    }

    // In oldchunk.rs

    pub fn from_bytes(bytes: [[u8; SIZE_X / 8]; SIZE_Y]) -> Self {
        let mut chunk = Self::empty();
        for r in 0..SIZE_Y {
            let y = SIZE_Y - 1 - r;
            for x in 0..SIZE_X {
                let bit = bytes[r][x / 8] & (0x80 >> (x % 8));
                chunk.data[x][y] = (bit != 0).into();
            }
        }
        chunk
    }

    pub fn toggle(&mut self, x: usize, y: usize) {
        self.data[x][y] ^= Cell::Alive;
    }

    pub fn is_active(&self) -> bool {
        is_active(&self.data)
    }

    pub fn will_be_active(&self) -> bool {
        is_active(&self.alternate)
    }

    fn apply_rules(&mut self, x: usize, y: usize, num_neighbours: u8) {
        self.alternate[x][y] =
            (num_neighbours == 3 || (num_neighbours == 2 && self.data[x][y] == Alive)).into();
    }

    pub fn check(
        &mut self,
        top: &Chunk,
        top_right: &Chunk,
        right: &Chunk,
        bottom_right: &Chunk,
        bottom: &Chunk,
        bottom_left: &Chunk,
        left: &Chunk,
        top_left: &Chunk,
    ) {
        self.alternate.fill([Dead; _]);

        for x in 1..SIZE_X - 1 {
            for y in 1..SIZE_Y - 1 {
                self.apply_rules(
                    x,
                    y,
                    self[(x - 1, y + 1)]
                        + self[(x, y + 1)]
                        + self[(x + 1, y + 1)]
                        + self[(x - 1, y)]
                        + self[(x + 1, y)]
                        + self[(x - 1, y - 1)]
                        + self[(x, y - 1)]
                        + self[(x + 1, y - 1)],
                );
            }
        }

        // top edge
        for x in 1..SIZE_X - 1 {
            self.apply_rules(
                x,
                SIZE_Y - 1,
                top[(x - 1, 0)]
                    + top[(x, 0)]
                    + top[(x + 1, 0)]
                    + self[(x - 1, SIZE_Y - 1)]
                    + self[(x + 1, SIZE_Y - 1)]
                    + self[(x - 1, SIZE_Y - 2)]
                    + self[(x, SIZE_Y - 2)]
                    + self[(x + 1, SIZE_Y - 2)],
            );
        }

        // top-right corner
        self.apply_rules(
            SIZE_X - 1,
            SIZE_Y - 1,
            top[(SIZE_X - 2, 0)]
                + top[(SIZE_X - 1, 0)]
                + top_right[(0, 0)]
                + self[(SIZE_X - 2, SIZE_Y - 1)]
                + right[(0, SIZE_Y - 1)]
                + self[(SIZE_X - 2, SIZE_Y - 2)]
                + self[(SIZE_X - 1, SIZE_Y - 2)]
                + right[(0, SIZE_Y - 2)],
        );

        // right edge
        for y in 1..SIZE_Y - 1 {
            self.apply_rules(
                SIZE_X - 1,
                y,
                self[(SIZE_X - 2, y + 1)]
                    + self[(SIZE_X - 1, y + 1)]
                    + right[(0, y + 1)]
                    + self[(SIZE_X - 2, y)]
                    + right[(0, y)]
                    + self[(SIZE_X - 2, y - 1)]
                    + self[(SIZE_X - 1, y - 1)]
                    + right[(0, y - 1)],
            );
        }

        // bottom-right corner
        self.apply_rules(
            SIZE_X - 1,
            0,
            self[(SIZE_X - 2, 1)]
                + self[(SIZE_X - 1, 1)]
                + right[(0, 1)]
                + self[(SIZE_X - 2, 0)]
                + right[(0, 0)]
                + bottom[(SIZE_X - 2, SIZE_Y - 1)]
                + bottom[(SIZE_X - 1, SIZE_Y - 1)]
                + bottom_right[(0, SIZE_Y - 1)],
        );

        // bottom edge
        for x in 1..SIZE_X - 1 {
            self.apply_rules(
                x,
                0,
                self[(x - 1, 1)]
                    + self[(x, 1)]
                    + self[(x + 1, 1)]
                    + self[(x - 1, 0)]
                    + self[(x + 1, 0)]
                    + bottom[(x - 1, SIZE_Y - 1)]
                    + bottom[(x, SIZE_Y - 1)]
                    + bottom[(x + 1, SIZE_Y - 1)],
            );
        }

        // bottom-left corner
        self.apply_rules(
            0,
            0,
            left[(SIZE_X - 1, 1)]
                + self[(0, 1)]
                + self[(1, 1)]
                + left[(SIZE_X - 1, 0)]
                + self[(1, 0)]
                + bottom_left[(SIZE_X - 1, SIZE_Y - 1)]
                + bottom[(0, SIZE_Y - 1)]
                + bottom[(1, SIZE_Y - 1)],
        );

        // left edge
        for y in 1..SIZE_Y - 1 {
            self.apply_rules(
                0,
                y,
                left[(SIZE_X - 1, y + 1)]
                    + self[(0, y + 1)]
                    + self[(1, y + 1)]
                    + left[(SIZE_X - 1, y)]
                    + self[(1, y)]
                    + left[(SIZE_X - 1, y - 1)]
                    + self[(0, y - 1)]
                    + self[(1, y - 1)],
            );
        }

        // top-left corner
        self.apply_rules(
            0,
            SIZE_Y - 1,
            top_left[(SIZE_X - 1, 0)]
                + top[(0, 0)]
                + top[(1, 0)]
                + left[(SIZE_X - 1, SIZE_Y - 1)]
                + self[(1, SIZE_Y - 1)]
                + left[(SIZE_X - 1, SIZE_Y - 2)]
                + self[(0, SIZE_Y - 2)]
                + self[(1, SIZE_Y - 2)],
        );
    }

    pub fn check_borders(
        top: &Chunk,
        top_right: &Chunk,
        right: &Chunk,
        bottom_right: &Chunk,
        bottom: &Chunk,
        bottom_left: &Chunk,
        left: &Chunk,
        top_left: &Chunk,
    ) -> Option<Chunk> {
        let mut output = [[Dead; _]; _];

        let mut apply_rules = |x: usize, y, num_neighbours| {
            output[x][y] = (num_neighbours == 3).into();
        };

        // top edge
        for x in 1..SIZE_X - 1 {
            apply_rules(
                x,
                SIZE_Y - 1,
                top[(x - 1, 0)] + top[(x, 0)] + top[(x + 1, 0)],
            );
        }

        // top-right corner
        apply_rules(
            SIZE_X - 1,
            SIZE_Y - 1,
            top[(SIZE_X - 2, 0)]
                + top[(SIZE_X - 1, 0)]
                + top_right[(0, 0)]
                + right[(0, SIZE_Y - 1)]
                + right[(0, SIZE_Y - 2)],
        );

        // right edge
        for y in 1..SIZE_Y - 1 {
            apply_rules(
                SIZE_X - 1,
                y,
                right[(0, y + 1)] + right[(0, y)] + right[(0, y - 1)],
            );
        }

        // bottom-right corner
        apply_rules(
            SIZE_X - 1,
            0,
            right[(0, 1)]
                + right[(0, 0)]
                + bottom[(SIZE_X - 2, SIZE_Y - 1)]
                + bottom[(SIZE_X - 1, SIZE_Y - 1)]
                + bottom_right[(0, SIZE_Y - 1)],
        );

        // bottom edge
        for x in 1..SIZE_X - 1 {
            apply_rules(
                x,
                0,
                bottom[(x - 1, SIZE_Y - 1)] + bottom[(x, SIZE_Y - 1)] + bottom[(x + 1, SIZE_Y - 1)],
            );
        }

        // bottom-left corner
        apply_rules(
            0,
            0,
            left[(SIZE_X - 1, 1)]
                + left[(SIZE_X - 1, 0)]
                + bottom_left[(SIZE_X - 1, SIZE_Y - 1)]
                + bottom[(0, SIZE_Y - 1)]
                + bottom[(1, SIZE_Y - 1)],
        );

        // left edge
        for y in 1..SIZE_Y - 1 {
            apply_rules(
                0,
                y,
                left[(SIZE_X - 1, y + 1)] + left[(SIZE_X - 1, y)] + left[(SIZE_X - 1, y - 1)],
            );
        }

        // top-left corner
        apply_rules(
            0,
            SIZE_Y - 1,
            top_left[(SIZE_X - 1, 0)]
                + top[(0, 0)]
                + top[(1, 0)]
                + left[(SIZE_X - 1, SIZE_Y - 1)]
                + left[(SIZE_X - 1, SIZE_Y - 2)],
        );

        if is_active(&output) {
            Some(Chunk {
                data: [[Dead; _]; _],
                alternate: output,
            })
        } else {
            None
        }
    }

    pub fn update(&mut self) {
        mem::swap(&mut self.alternate, &mut self.data);
    }
}

impl Display for Chunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in (0..SIZE_Y).rev() {
            f.write_char('\n')?;
            for x in 0..SIZE_X {
                f.write_char(match self[(x, y)] {
                    Dead => '.',
                    Alive => '#',
                })?
            }
        }
        Ok(())
    }
}

impl Debug for Chunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self, f)
    }
}
