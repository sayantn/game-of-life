use std::fmt;

use crate::traits::Chunk;

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

pub fn parse_chunk<C: Chunk>(s: &str) -> Result<C, ParseError> {
    let mut bytes = C::empty().to_bytes();

    let buf = bytes.as_mut();
    debug_assert_eq!(buf.len(), C::BYTES);

    let lines: Vec<&str> = s.lines().collect();

    if lines.len() > C::SIZE {
        return Err(ParseError::WrongRowCount {
            expected: C::SIZE,
            found: lines.len(),
        });
    }

    // Every row must have this width. An empty input has width 0.
    let width = lines.first().map_or(0, |line| line.chars().count());
    if width > C::SIZE {
        return Err(ParseError::WrongRowLength {
            row: 0,
            expected: C::SIZE,
            found: width,
        });
    }

    for (r, (line, row)) in lines
        .iter()
        .zip(buf.chunks_exact_mut(C::SIZE / 8))
        .enumerate()
    {
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
                '#' => row[col / 8] |= 0x80 >> (col % 8),
                '.' => {}
                found => return Err(ParseError::InvalidCell { row: r, col, found }),
            }
        }
    }

    Ok(C::from(bytes))
}

pub fn print_chunk<C: Chunk>(f: &mut fmt::Formatter<'_>, chunk: &C) -> fmt::Result {
    let mut line = vec![b'\n'; C::SIZE + 1];

    let bytes = chunk.to_bytes();
    let bytes = bytes.as_ref();

    for row in bytes.chunks_exact(C::SIZE / 8) {
        for (j, slot) in line[..C::SIZE].iter_mut().enumerate() {
            let bit = row[j / 8] & (0x80 >> (j % 8));
            *slot = if bit != 0 { b'#' } else { b'.' };
        }
        f.write_str(str::from_utf8(&line).unwrap())?;
    }

    Ok(())
}
