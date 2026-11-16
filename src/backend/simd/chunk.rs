use std::fmt::Debug;
use std::iter::zip;

use crate::{Activity, Chunk, Dir, NeighborData, Orth};

use super::row::*;

const SIZE: usize = ROW_BITS;

const _: () = assert!(ROW_BITS == Chunk::SIZE);

#[derive(Clone, PartialEq, Eq)]
pub(super) struct SimdChunk {
    data: [Row; ROW_BITS],
}

impl SimdChunk {
    #[allow(clippy::large_types_passed_by_value)]
    const fn new(data: [Row; ROW_BITS]) -> Self {
        Self { data }
    }
}

impl SimdChunk {
    #[inline]
    pub(super) const fn empty() -> Self {
        Self::new([empty_row(); _])
    }

    pub(super) fn convert_into(&self, dst: &mut Chunk) {
        for (src, dst) in zip(self.data, dst.as_rows_mut()) {
            *dst = row_to_bytes(src);
        }
    }
}

impl Chunk {
    pub(super) fn convert_into(&self, dst: &mut SimdChunk) {
        for (src, dst) in zip(self.as_rows(), &mut dst.data) {
            *dst = row_from_bytes(*src);
        }
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
