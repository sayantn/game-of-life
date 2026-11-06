use std::fmt::{self, Debug};
use std::iter;
use std::str::FromStr;

use rand::Rng;

use crate::row::*;

pub const SIZE: usize = 8 * ROW_BYTES;

#[derive(Clone, PartialEq, Eq)]
pub struct Chunk {
    data: [Row; SIZE],
}

impl Chunk {
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
        self.data.iter().any(|&row| row != empty_row())
    }

    pub fn apply_rules(&self, nbhd: Neighbourhood<'_>, dst: &mut Self) {
        let mut top_row = RowDigest::new(
            nbhd.n.data[SIZE - 1],
            nbhd.nw.data[SIZE - 1],
            nbhd.ne.data[SIZE - 1],
        );
        let mut current_row = RowDigest::new(self.data[0], nbhd.w.data[0], nbhd.e.data[0]);

        for (i, bottom_row) in (1..SIZE)
            .map(|i| (self.data[i], nbhd.w.data[i], nbhd.e.data[i]))
            .chain(iter::once((
                nbhd.s.data[0],
                nbhd.sw.data[0],
                nbhd.se.data[0],
            )))
            .map(|(row, left, right)| RowDigest::new(row, left, right))
            .enumerate()
        {
            dst.data[i] = rule(
                top_row.triplet,
                current_row.doublet,
                bottom_row.triplet,
                current_row.row,
            );

            (top_row, current_row) = (current_row, bottom_row);
        }
    }

    pub fn check_activity(_nbhd: &Neighbourhood<'_>, _dst: &mut Self) {
        todo!()
    }
}

impl Debug for Chunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut line = [b'\n'; SIZE + 1];
        for row in self.data {
            let bytes = row_to_bytes(row);
            for (j, slot) in line[1..].iter_mut().enumerate() {
                let bit = bytes[j / 8] & (0x80 >> (j % 8));
                *slot = if bit != 0 { b'#' } else { b'.' };
            }
            f.write_str(std::str::from_utf8(&line).unwrap())?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Neighbourhood<'a> {
    pub nw: &'a Chunk,
    pub n: &'a Chunk,
    pub ne: &'a Chunk,
    pub w: &'a Chunk,
    pub e: &'a Chunk,
    pub sw: &'a Chunk,
    pub s: &'a Chunk,
    pub se: &'a Chunk,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    WrongRowCount {
        expected: usize,
        found: usize,
    },
    WrongRowLength {
        row: usize,
        expected: usize,
        found: usize,
    },
    RaggedRow {
        row: usize,
        expected: usize,
        found: usize,
    },
    InvalidCell {
        row: usize,
        col: usize,
        found: char,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::WrongRowCount { expected, found } => {
                write!(f, "expected at most {expected} rows, found {found}")
            }
            Self::WrongRowLength {
                row,
                expected,
                found,
            } => {
                write!(
                    f,
                    "row {row}: expected at most {expected} cells, found {found}"
                )
            }
            Self::RaggedRow {
                row,
                expected,
                found,
            } => {
                write!(f, "row {row}: expected {expected} cells, found {found}")
            }
            Self::InvalidCell { row, col, found } => {
                write!(f, "row {row}, column {col}: invalid cell {found:?}")
            }
        }
    }
}

impl std::error::Error for ParseError {}

impl FromStr for Chunk {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lines: Vec<&str> = s.lines().collect();

        if lines.len() > SIZE {
            return Err(ParseError::WrongRowCount {
                expected: SIZE,
                found: lines.len(),
            });
        }

        // Every row must have this width. An empty input has width 0.
        let width = lines.first().map_or(0, |l| l.chars().count());
        if width > SIZE {
            return Err(ParseError::WrongRowLength {
                row: 0,
                expected: SIZE,
                found: width,
            });
        }

        // Zero-initialised, so rows and columns outside the input stay dead.
        let mut bytes = [[0u8; ROW_BYTES]; SIZE];

        for (r, line) in lines.iter().enumerate() {
            // Check the width before indexing, so `col < width <= COLS` holds below.
            let len = line.chars().count();
            if len != width {
                return Err(ParseError::RaggedRow {
                    row: r,
                    expected: width,
                    found: len,
                });
            }

            for (col, c) in line.chars().enumerate() {
                match c {
                    '#' => bytes[r][col / 8] |= 0x80 >> (col % 8),
                    '.' => {}
                    found => return Err(ParseError::InvalidCell { row: r, col, found }),
                }
            }
        }

        Ok(Self::from_bytes(bytes))
    }
}
