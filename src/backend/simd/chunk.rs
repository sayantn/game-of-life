use std::iter::zip;

use crate::{Activity, Chunk, Dir, NeighborData, Orth};

use super::row::*;

pub(crate) const REGS_PER_CHUNK: usize = 256 / ROWS_PER_REG;

#[derive(Clone, PartialEq, Eq)]
pub(super) struct SimdChunk {
    data: [Reg; REGS_PER_CHUNK],
}

impl SimdChunk {
    #[allow(clippy::large_types_passed_by_value)]
    const fn new(data: [Reg; REGS_PER_CHUNK]) -> Self {
        Self { data }
    }

    #[inline]
    pub(super) const fn empty() -> Self {
        Self::new([empty_reg(); REGS_PER_CHUNK])
    }

    #[inline]
    pub(super) fn convert_into(&self, dst: &mut Chunk) {
        for (src, dst_bytes) in zip(self.data, dst.as_bytes_mut().chunks_exact_mut(REG_BYTES)) {
            dst_bytes.copy_from_slice(&reg_to_bytes(src));
        }
    }
}

impl Chunk {
    #[inline]
    pub(super) fn convert_into(&self, dst: &mut SimdChunk) {
        for (src_bytes, dst_reg) in zip(self.as_bytes().chunks_exact(REG_BYTES), &mut dst.data) {
            *dst_reg = reg_from_bytes(src_bytes.try_into().unwrap());
        }
    }
}

#[inline]
fn compute_activity(acc: Reg, top_reg: Reg, bottom_reg: Reg) -> SimdActivity {
    SimdActivity {
        chunk: acc != empty_reg(),
        top: is_top_row_active(top_reg),
        bottom: is_bottom_row_active(bottom_reg),
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

// Applies vertical lane shifts directly to the cached sums to avoid redundant bitwise math
#[inline]
fn shift_digest_north(curr: RegDigest, south: RegDigest) -> RegDigest {
    RegDigest {
        reg: shift_regs_north(curr.reg, south.reg),
        doublet: (
            shift_regs_north(curr.doublet.0, south.doublet.0),
            shift_regs_north(curr.doublet.1, south.doublet.1),
        ),
        triplet: (
            shift_regs_north(curr.triplet.0, south.triplet.0),
            shift_regs_north(curr.triplet.1, south.triplet.1),
        ),
    }
}

#[inline]
fn shift_digest_south(curr: RegDigest, north: RegDigest) -> RegDigest {
    RegDigest {
        reg: shift_regs_south(curr.reg, north.reg),
        doublet: (
            shift_regs_south(curr.doublet.0, north.doublet.0),
            shift_regs_south(curr.doublet.1, north.doublet.1),
        ),
        triplet: (
            shift_regs_south(curr.triplet.0, north.triplet.0),
            shift_regs_south(curr.triplet.1, north.triplet.1),
        ),
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

    let mut prev_digest = RegDigest::new(
        nbhd[Dir::N].data[REGS_PER_CHUNK - 1],
        nbhd[Dir::NW].data[REGS_PER_CHUNK - 1],
        nbhd[Dir::NE].data[REGS_PER_CHUNK - 1],
    );
    let mut curr_digest = RegDigest::new(center.data[0], left[0], right[0]);

    let mut acc = empty_reg();

    for i in 0..REGS_PER_CHUNK {
        let next_center = if i + 1 < REGS_PER_CHUNK {
            center.data[i + 1]
        } else {
            nbhd[Dir::S].data[0]
        };
        let next_left = if i + 1 < REGS_PER_CHUNK {
            left[i + 1]
        } else {
            nbhd[Dir::SW].data[0]
        };
        let next_right = if i + 1 < REGS_PER_CHUNK {
            right[i + 1]
        } else {
            nbhd[Dir::SE].data[0]
        };

        // 1 digest calculation per loop instead of 3
        let next_digest = RegDigest::new(next_center, next_left, next_right);

        let top_digest = shift_digest_south(curr_digest, prev_digest);
        let bottom_digest = shift_digest_north(curr_digest, next_digest);

        let result = rule(
            top_digest.triplet,
            curr_digest.doublet,
            bottom_digest.triplet,
            curr_digest.reg,
        );

        dst.data[i] = result;
        acc |= result;

        prev_digest = curr_digest;
        curr_digest = next_digest;
    }

    compute_activity(acc, dst.data[0], dst.data[REGS_PER_CHUNK - 1])
}

#[inline]
pub(super) fn check_activity(nbhd: NeighborData<&SimdChunk>, dst: &mut SimdChunk) -> SimdActivity {
    let left = &nbhd[Dir::W].data;
    let right = &nbhd[Dir::E].data;

    let mut prev_digest = RegDigest::new(
        nbhd[Dir::N].data[REGS_PER_CHUNK - 1],
        nbhd[Dir::NW].data[REGS_PER_CHUNK - 1],
        nbhd[Dir::NE].data[REGS_PER_CHUNK - 1],
    );
    let mut curr_digest = RegDigest::new(empty_reg(), left[0], right[0]);

    let mut acc = empty_reg();

    for i in 0..REGS_PER_CHUNK {
        // FIXED: Next center correctly pulls from the South neighbor's top row at the boundary
        let next_center = if i + 1 < REGS_PER_CHUNK {
            empty_reg()
        } else {
            nbhd[Dir::S].data[0]
        };
        let next_left = if i + 1 < REGS_PER_CHUNK {
            left[i + 1]
        } else {
            nbhd[Dir::SW].data[0]
        };
        let next_right = if i + 1 < REGS_PER_CHUNK {
            right[i + 1]
        } else {
            nbhd[Dir::SE].data[0]
        };

        let next_digest = RegDigest::new(next_center, next_left, next_right);

        let top_digest = shift_digest_south(curr_digest, prev_digest);
        let bottom_digest = shift_digest_north(curr_digest, next_digest);

        let result = rule(
            top_digest.triplet,
            curr_digest.doublet,
            bottom_digest.triplet,
            curr_digest.reg, // Always evaluates to empty_reg() for vacant chunks
        );

        dst.data[i] = result;
        acc |= result;

        prev_digest = curr_digest;
        curr_digest = next_digest;
    }

    compute_activity(acc, dst.data[0], dst.data[REGS_PER_CHUNK - 1])
}
