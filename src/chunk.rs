use std::fmt::{self, Debug};
use std::ops::{BitOr, Index, IndexMut};

use rand::Rng;

use crate::row::*;

pub const SIZE: usize = ROW_BITS;

#[derive(Clone, PartialEq, Eq)]
pub struct Chunk {
    data: [Row; SIZE],
}

impl Chunk {
    #[allow(clippy::large_types_passed_by_value)]
    fn new(data: [Row; SIZE]) -> Self {
        Self { data }
    }

    pub fn empty() -> Self {
        Self::new([empty_row(); _])
    }

    pub fn from_bytes(bytes: [[u8; ROW_BYTES]; SIZE]) -> Self {
        Self::new(bytes.map(row_from_bytes))
    }

    pub fn random(rng: &mut impl Rng) -> Self {
        Self::from_bytes(rng.random())
    }

    pub fn to_bytes(&self) -> [[u8; ROW_BYTES]; SIZE] {
        self.data.map(row_to_bytes)
    }

    pub fn num_alive(&self) -> usize {
        self.data.iter().copied().map(row_num_alive).sum()
    }

    pub fn is_empty(&self) -> bool {
        for block in self.data.as_chunks::<16>().0 {
            let acc = unsafe { block.iter().copied().reduce(Row::bitor).unwrap_unchecked() };
            if acc != empty_row() {
                return false;
            }
        }
        true
    }

    pub fn apply_rules(&self, nbhd: NeighborData<&Chunk>, dst: &mut Self) -> Activity {
        let left = &nbhd[Dir::W].data;
        let right = &nbhd[Dir::E].data;

        let mut top_row = RowDigest::new(
            nbhd[Dir::N].data[SIZE - 1],
            nbhd[Dir::NW].data[SIZE - 1],
            nbhd[Dir::NE].data[SIZE - 1],
        );
        let mut current_row = RowDigest::new(self.data[0], left[0], right[0]);

        let mut acc = empty_row();

        for i in 0..SIZE - 1 {
            let bottom_row = RowDigest::new(self.data[i + 1], left[i + 1], right[i + 1]);
            let result = rule(
                top_row.triplet,
                current_row.doublet,
                bottom_row.triplet,
                current_row.row,
            );
            dst.data[i] = result;

            acc |= result;

            top_row = current_row;
            current_row = bottom_row;
        }

        let bottom_row = RowDigest::new(
            nbhd[Dir::S].data[0],
            nbhd[Dir::SW].data[0],
            nbhd[Dir::SE].data[0],
        );
        let result = rule(
            top_row.triplet,
            current_row.doublet,
            bottom_row.triplet,
            current_row.row,
        );
        dst.data[SIZE - 1] = result;

        acc |= result;

        compute_activity(acc, dst.data[0], result)
    }

    pub fn check_activity(nbhd: NeighborData<&Chunk>, dst: &mut Self) -> Activity {
        let left = &nbhd[Dir::W].data;
        let right = &nbhd[Dir::E].data;

        let mut top_row = RowDigest::new(
            nbhd[Dir::N].data[SIZE - 1],
            nbhd[Dir::NW].data[SIZE - 1],
            nbhd[Dir::NE].data[SIZE - 1],
        );
        let mut current_row = RowDigest::new(empty_row(), left[0], right[0]);
        let mut acc = empty_row();

        for i in 0..SIZE - 1 {
            let bottom_row = RowDigest::new(empty_row(), left[i + 1], right[i + 1]);
            let result = rule(
                top_row.triplet,
                current_row.doublet,
                bottom_row.triplet,
                current_row.row,
            );
            dst.data[i] = result;
            acc |= result;
            top_row = current_row;
            current_row = bottom_row;
        }

        let bottom_row = RowDigest::new(
            nbhd[Dir::S].data[0],
            nbhd[Dir::SW].data[0],
            nbhd[Dir::SE].data[0],
        );
        let result = rule(
            top_row.triplet,
            current_row.doublet,
            bottom_row.triplet,
            current_row.row,
        );
        dst.data[SIZE - 1] = result;
        acc |= result;

        compute_activity(acc, dst.data[0], result)
    }
}

impl Debug for Chunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut line = [b'\n'; SIZE + 1];
        for row in self.data {
            let bytes = row_to_bytes(row);
            for (j, slot) in line[..SIZE].iter_mut().enumerate() {
                let bit = bytes[j / 8] & (0x80 >> (j % 8));
                *slot = if bit != 0 { b'#' } else { b'.' };
            }
            f.write_str(str::from_utf8(&line).unwrap())?;
        }
        Ok(())
    }
}

fn compute_activity(acc: Row, top_row: Row, bottom_row: Row) -> Activity {
    Activity {
        chunk: acc != empty_row(),
        top: top_row != empty_row(),
        bottom: bottom_row != empty_row(),
        left_right: edge_cols(acc),
    }
}

#[derive(Clone, Copy)]
pub struct Activity {
    left_right: Lane,
    chunk: bool,
    top: bool,
    bottom: bool,
}

impl Activity {
    pub const ALL: Activity = Activity {
        left_right: !0,
        chunk: true,
        top: true,
        bottom: true,
    };

    pub const EMPTY: Activity = Activity {
        left_right: 0,
        chunk: false,
        top: false,
        bottom: false,
    };

    pub fn chunk(self) -> bool {
        self.chunk
    }

    pub fn edge(self, orth: Orth) -> bool {
        match orth {
            Orth::N => self.top,
            Orth::W => self.left_right >> 1 != 0,
            Orth::E => self.left_right << 1 != 0,
            Orth::S => self.bottom,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orth {
    N,
    W,
    E,
    S,
}

impl Orth {
    pub const ALL: [Orth; 4] = [Orth::N, Orth::W, Orth::E, Orth::S];

    pub fn dir(self) -> Dir {
        match self {
            Self::N => Dir::N,
            Self::W => Dir::W,
            Self::E => Dir::E,
            Self::S => Dir::S,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    NW,
    N,
    NE,
    W,
    E,
    SW,
    S,
    SE,
}

impl Dir {
    pub const ALL: [Dir; 8] = [
        Dir::NW,
        Dir::N,
        Dir::NE,
        Dir::W,
        Dir::E,
        Dir::SW,
        Dir::S,
        Dir::SE,
    ];

    #[inline]
    pub fn opposite(self) -> Dir {
        Dir::ALL[7 - self as usize]
    }

    #[inline]
    pub fn offset(self) -> (i64, i64) {
        match self {
            Dir::NW => (-1, 1),
            Dir::N => (0, 1),
            Dir::NE => (1, 1),
            Dir::W => (-1, 0),
            Dir::E => (1, 0),
            Dir::SW => (-1, -1),
            Dir::S => (0, -1),
            Dir::SE => (1, -1),
        }
    }

    #[inline]
    pub fn from_offset(dx: i64, dy: i64) -> Self {
        match (dx, dy) {
            (-1, 1) => Dir::NW,
            (0, 1) => Dir::N,
            (1, 1) => Dir::NE,
            (-1, 0) => Dir::W,
            (1, 0) => Dir::E,
            (-1, -1) => Dir::SW,
            (0, -1) => Dir::S,
            (1, -1) => Dir::SE,
            _ => panic!("({dx}, {dy}) is not a direction"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighborData<T>([T; 8]);

impl<T> Index<Dir> for NeighborData<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: Dir) -> &Self::Output {
        &self.0[index as usize]
    }
}

impl<T> IndexMut<Dir> for NeighborData<T> {
    #[inline]
    fn index_mut(&mut self, index: Dir) -> &mut Self::Output {
        &mut self.0[index as usize]
    }
}

impl<T> NeighborData<T> {
    pub fn from_fn(f: impl FnMut(Dir) -> T) -> Self {
        Self(Dir::ALL.map(f))
    }

    pub fn as_ref(&self) -> NeighborData<&T> {
        NeighborData(self.0.each_ref())
    }

    pub fn map<U>(self, f: impl FnMut(T) -> U) -> NeighborData<U> {
        NeighborData(self.0.map(f))
    }

    pub fn iter(&self) -> impl Iterator<Item = (Dir, &T)> {
        Dir::ALL.into_iter().zip(&self.0)
    }
}

#[cfg(test)]
mod chunk_compat_tests {
    use rand::{SeedableRng, rngs::StdRng};

    use crate::chunk::{Chunk as NewChunk, Dir, NeighborData};
    use crate::naive::chunk::Chunk as OldChunk;

    #[test]
    fn test_chunk_bitwise_rules_match_naive() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..50 {
            let new_center = NewChunk::random(&mut rng);
            let mut old_center = OldChunk::from_bytes(new_center.to_bytes());

            let new_nbhd = NeighborData::from_fn(|_| NewChunk::random(&mut rng));

            let old_nbhd = new_nbhd
                .as_ref()
                .map(|nc| OldChunk::from_bytes(nc.to_bytes()));

            old_center.check(
                &old_nbhd[Dir::N],
                &old_nbhd[Dir::NE],
                &old_nbhd[Dir::E],
                &old_nbhd[Dir::SE],
                &old_nbhd[Dir::S],
                &old_nbhd[Dir::SW],
                &old_nbhd[Dir::W],
                &old_nbhd[Dir::NW],
            );
            old_center.update();

            let mut dst = NewChunk::empty();
            new_center.apply_rules(NeighborData::from_fn(|dir| &new_nbhd[dir]), &mut dst);

            let old_from_new = OldChunk::from_bytes(dst.to_bytes());
            assert_eq!(old_center, old_from_new);
        }
    }
}
