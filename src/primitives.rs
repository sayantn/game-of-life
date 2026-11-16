use std::ops::{Index, IndexMut};

/// Cardinal (orthogonal) direction corresponding to the four borders of a square chunk.
#[repr(u8)]
#[must_use]
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

/// Eight-way compass direction (orthogonal and diagonal) to an adjacent chunk.
#[repr(u8)]
#[must_use]
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
    #[must_use]
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

/// Fixed-size container associating a value of type `T` with each of the 8 [`Dir`] neighbors.
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

/// 2D chunk-space grid coordinate `(x, y)` in an [`InfiniteFlatWorld`](crate::InfiniteFlatWorld).
#[must_use]
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
