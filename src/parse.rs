use std::fmt;
use std::str::FromStr;

use crate::chunk::{Chunk, SIZE};
use crate::row::ROW_BYTES;

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
