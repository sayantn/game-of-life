use std::fmt::{self, Debug};
use std::mem;
use std::ops::BitOr;
use std::str::FromStr;

use crate::{Activity, Chunk, Dir, NeighborData, Orth, helpers};

use super::row::*;

const SIZE: usize = ROW_BITS;

#[derive(Clone, PartialEq, Eq)]
pub struct SimdChunk {
    data: [Row; ROW_BITS],
}

impl SimdChunk {
    #[allow(clippy::large_types_passed_by_value)]
    fn new(data: [Row; ROW_BITS]) -> Self {
        Self { data }
    }
}

impl Chunk for SimdChunk {
    const SIZE: usize = ROW_BITS;

    type Bytes = [u8; ROW_BYTES * ROW_BITS];

    #[inline]
    fn empty() -> Self {
        Self::new([empty_row(); _])
    }

    #[inline]
    fn from_bytes(bytes: &[u8; ROW_BYTES * ROW_BITS]) -> Self {
        Self::from(*bytes)
    }

    #[inline]
    fn to_bytes(&self) -> [u8; ROW_BYTES * ROW_BITS] {
        self.clone().into()
    }

    #[inline]
    fn num_alive(&self) -> usize {
        self.data.iter().copied().map(row_num_alive).sum()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        for block in self.data.as_chunks::<16>().0 {
            // SAFETY: Each chunk from `as_chunks::<16>()` has length 16, so the iterator is non-empty.
            let acc = unsafe { block.iter().copied().reduce(Row::bitor).unwrap_unchecked() };
            if acc != empty_row() {
                return false;
            }
        }
        true
    }
}

impl Default for SimdChunk {
    #[inline]
    fn default() -> Self {
        Self {
            data: [empty_row(); _],
        }
    }
}

impl From<[u8; ROW_BYTES * ROW_BITS]> for SimdChunk {
    #[inline]
    fn from(value: [u8; ROW_BYTES * ROW_BITS]) -> Self {
        // SAFETY: `[u8; ROW_BYTES * ROW_BITS]` and `[[u8; ROW_BYTES]; ROW_BITS]` have identical
        // size, alignment (1), and bit validity.
        let bytes: [[u8; ROW_BYTES]; ROW_BITS] = unsafe { mem::transmute(value) };
        Self::new(bytes.map(row_from_bytes))
    }
}

impl From<SimdChunk> for [u8; ROW_BYTES * ROW_BITS] {
    #[inline]
    fn from(value: SimdChunk) -> Self {
        // SAFETY: `[[u8; ROW_BYTES]; ROW_BITS]` and `[u8; ROW_BYTES * ROW_BITS]` have identical
        // size, alignment (1), and bit validity.
        unsafe { mem::transmute(value.data.map(row_to_bytes)) }
    }
}

impl Debug for SimdChunk {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        helpers::print_chunk(f, self)
    }
}

impl FromStr for SimdChunk {
    type Err = helpers::ParseError;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        helpers::parse_chunk(s)
    }
}

fn compute_activity(acc: Row, top_row: Row, bottom_row: Row) -> SimdActivity {
    SimdActivity {
        chunk: acc != empty_row(),
        top: top_row != empty_row(),
        bottom: bottom_row != empty_row(),
        left_right: edge_cols(acc),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdActivity {
    left_right: Lane,
    chunk: bool,
    top: bool,
    bottom: bool,
}

impl Activity for SimdActivity {
    const ALL: SimdActivity = SimdActivity {
        left_right: !0,
        chunk: true,
        top: true,
        bottom: true,
    };

    const EMPTY: SimdActivity = SimdActivity {
        left_right: 0,
        chunk: false,
        top: false,
        bottom: false,
    };

    #[inline]
    fn chunk(self) -> bool {
        self.chunk
    }

    #[inline]
    fn edge(self, orth: Orth) -> bool {
        match orth {
            Orth::N => self.top,
            Orth::W => self.left_right >> 1 != 0,
            Orth::E => self.left_right << 1 != 0,
            Orth::S => self.bottom,
        }
    }
}

#[inline]
pub(super) fn apply_rules(
    center: &SimdChunk,
    nbhd: NeighborData<&SimdChunk>,
    dst: &mut SimdChunk,
) -> SimdActivity {
    let left = &nbhd[Dir::W].data;
    let right = &nbhd[Dir::E].data;

    let mut top_row = RowDigest::new(
        nbhd[Dir::N].data[SIZE - 1],
        nbhd[Dir::NW].data[SIZE - 1],
        nbhd[Dir::NE].data[SIZE - 1],
    );
    let mut current_row = RowDigest::new(center.data[0], left[0], right[0]);

    let mut acc = empty_row();

    for i in 0..SIZE - 1 {
        let bottom_row = RowDigest::new(center.data[i + 1], left[i + 1], right[i + 1]);
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

#[inline]
pub(super) fn check_activity(nbhd: NeighborData<&SimdChunk>, dst: &mut SimdChunk) -> SimdActivity {
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
