use std::str::FromStr;

use crate::helpers::{ParseError, parse_chunk, print_chunk};

/// Canonical, architecture-independent 256×256 Game of Life chunk.
///
/// Serves as the universal data currency across all [`Buffer`](crate::Buffer)
/// backends (CPU scalar, SIMD, and GPU). Each backend converts between `Chunk`
/// and its internal memory layout during [`load`](crate::Buffer::load)
/// and [`store`](crate::Buffer::store).
///
/// # Memory Layout
/// - **Dimensions:** 256 rows (`y` in `0..256`) × 256 columns (`x` in `0..256`),
///   totaling 65,536 cells (8,192 bytes).
/// - **Row-major storage:** Row `y` occupies bytes `[y * 32 .. (y + 1) * 32]`.
/// - **MSB-first bit ordering:** Within byte `x / 8` of a row, column `x` is
///   stored at bit `7 - (x % 8)` (`0x80 >> (x & 7)`), so bit 7 (MSB) is the
///   leftmost cell (`x % 8 == 0`) and bit 0 (LSB) is the rightmost cell
///   (`x % 8 == 7`).
/// - **Alignment:** 64-byte aligned to match cache lines and AVX-512 vector
///   register boundaries for fast unaligned/aligned gateway loads and stores.
#[repr(C, align(64))]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Chunk([u8; Self::BYTES]);

impl Chunk {
    /// The width and height of the chunk in cells. MUST be 256.
    pub const SIZE: usize = 256;
    /// The number of bytes required to store a single row (32 bytes).
    pub const ROW_BYTES: usize = Self::SIZE / 8;
    /// The total number of bytes required to store the chunk (8192 bytes).
    pub const BYTES: usize = Self::SIZE * Self::ROW_BYTES;

    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self([0; Self::BYTES])
    }

    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; Self::BYTES]) -> Self {
        Self(bytes)
    }

    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; Self::BYTES] {
        self.0
    }

    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; Self::BYTES] {
        &self.0
    }

    #[inline]
    pub const fn as_bytes_mut(&mut self) -> &mut [u8; Self::BYTES] {
        &mut self.0
    }

    #[inline]
    #[must_use]
    pub const fn as_rows(&self) -> &[[u8; Self::ROW_BYTES]; Self::SIZE] {
        // SAFETY: `Chunk` has size `HEIGHT * ROW_BYTES == BYTES` and alignment 64 >= 1.
        unsafe { &*self.0.as_ptr().cast() }
    }

    #[inline]
    pub const fn as_rows_mut(&mut self) -> &mut [[u8; Self::ROW_BYTES]; Self::SIZE] {
        // SAFETY: `Chunk` has size `HEIGHT * ROW_BYTES == BYTES` and alignment 64 >= 1.
        unsafe { &mut *self.0.as_mut_ptr().cast() }
    }

    /// Returns `true` if the cell at `(x, y)` is alive.
    ///
    /// # Panics
    /// Panics if `x >= 256` or `y >= 256`.
    #[inline]
    #[must_use]
    pub const fn get(&self, x: usize, y: usize) -> bool {
        assert!(
            x < Self::SIZE && y < Self::SIZE,
            "cell coordinates out of bounds"
        );
        let byte = self.0[y * Self::ROW_BYTES + (x >> 3)];
        let mask = 0x80 >> (x & 7);
        (byte & mask) != 0
    }

    /// Sets the state of the cell at `(x, y)`.
    ///
    /// # Panics
    /// Panics if `x >= 256` or `y >= 256`.
    #[inline]
    pub const fn set(&mut self, x: usize, y: usize, alive: bool) {
        assert!(
            x < Self::SIZE && y < Self::SIZE,
            "cell coordinates out of bounds"
        );
        let idx = y * Self::ROW_BYTES + (x >> 3);
        let mask = 0x80 >> (x & 7);
        if alive {
            self.0[idx] |= mask;
        } else {
            self.0[idx] &= !mask;
        }
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }

    #[inline]
    #[must_use]
    pub fn population(&self) -> u32 {
        // SAFETY: `Chunk` is 64-byte aligned and 8192 bytes = 1024 `u64` words.
        let (prefix, words, suffix) = unsafe { self.0.align_to::<u64>() };
        debug_assert!(prefix.is_empty() && suffix.is_empty());
        words.iter().map(|w| w.count_ones()).sum()
    }
}

impl Default for Chunk {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

impl FromStr for Chunk {
    type Err = ParseError;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_chunk(s)
    }
}

impl std::fmt::Debug for Chunk {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // If the alternate flag is used ({:#?}), print the population summary instead
        // of blasting a 256x256 grid to the terminal.
        if f.alternate() {
            print_chunk(f, self)
        } else {
            f.debug_struct("Chunk")
                .field("population", &self.population())
                .finish()
        }
    }
}
