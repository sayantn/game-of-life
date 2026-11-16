use std::fmt;

use crate::Chunk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    #[inline]
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

#[inline]
pub(crate) fn parse_chunk(s: &str) -> Result<Chunk, ParseError> {
    let mut bytes = Chunk::empty().to_bytes();

    let buf = bytes.as_mut();
    debug_assert_eq!(buf.len(), Chunk::BYTES);

    let lines: Vec<&str> = s.lines().collect();

    if lines.len() > Chunk::SIZE {
        return Err(ParseError::WrongRowCount {
            expected: Chunk::SIZE,
            found: lines.len(),
        });
    }

    // Every row must have this width. An empty input has width 0.
    let width = lines.first().map_or(0, |line| line.chars().count());
    if width > Chunk::SIZE {
        return Err(ParseError::WrongRowLength {
            row: 0,
            expected: Chunk::SIZE,
            found: width,
        });
    }

    for (r, (line, row)) in lines
        .iter()
        .zip(buf.chunks_exact_mut(Chunk::SIZE / 8))
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

    Ok(Chunk::from_bytes(bytes))
}

#[inline]
pub(crate) fn print_chunk(f: &mut fmt::Formatter<'_>, chunk: &Chunk) -> fmt::Result {
    let mut line = vec![b'\n'; Chunk::SIZE + 1];

    let bytes = chunk.to_bytes();
    let bytes = bytes.as_ref();

    for row in bytes.chunks_exact(Chunk::SIZE / 8) {
        for (j, slot) in line[..Chunk::SIZE].iter_mut().enumerate() {
            let bit = row[j / 8] & (0x80 >> (j % 8));
            *slot = if bit != 0 { b'#' } else { b'.' };
        }
        f.write_str(str::from_utf8(&line).unwrap())?;
    }

    Ok(())
}

/// Reinterprets a vector's backing allocation as a different type.
///
/// # Safety (Internal)
/// Statically asserts that `T` and `U` have identical size and alignment constraints.
/// Clears the vector to length 0 before transmutation to prevent dropping misaligned types.
pub(crate) fn recycle_vec<T, U>(mut a: Vec<T>) -> Vec<U> {
    const {
        assert!(size_of::<T>() == size_of::<U>());
        assert!(align_of::<T>() == align_of::<U>());
    };

    a.clear();
    let (ptr, len, cap) = a.into_raw_parts();
    // SAFETY: `a` has length 0 (no initialized elements to drop or reinterpret), `T` and `U`
    // have identical size and alignment, and `(ptr, 0, cap)` came directly from `Vec::into_raw_parts`.
    unsafe { Vec::from_raw_parts(ptr.cast(), len, cap) }
}

/// Pushes an element into a vector, bypassing reallocation branches.
///
/// # Panics
/// Panics in debug builds if `vec.len() >= vec.capacity()`.
///
/// # Safety (Internal)
/// In release builds, silently drops the value if capacity is exceeded. If space exists,
/// writes directly to the spare capacity pointer, ensuring no uninitialized memory is read.
#[inline]
pub(crate) fn push_within_capacity<T>(vec: &mut Vec<T>, value: T) {
    debug_assert!(
        vec.len() < vec.capacity(),
        "push_within_capacity exceeded capacity!"
    );

    if vec.len() < vec.capacity() {
        // SAFETY: We verified len < capacity, so `as_mut_ptr().add(len)` is valid
        // for writes and within the allocation. We update the length immediately after.
        unsafe {
            let end = vec.as_mut_ptr().add(vec.len());
            end.write(value);
            vec.set_len(vec.len() + 1);
        }
    }
}
