use std::simd::*;

pub(super) type Reg = cfg_select! {
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ) => {
        cfg_select! {
            target_feature = "avx512f" => u64x8,
            _ => u64x4,
        }
    }

    all(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        target_feature = "neon"
    ) => u64x4,

    all(
        target_arch = "arm",
        any(target_feature = "neon", target_feature = "mve")
    ) => u32x8,

    all(
        any(target_arch = "powerpc", target_arch = "powerpc64"),
        target_feature = "altivec"
    ) => {
        cfg_select! {
            all(target_endian = "big", target_feature = "vsx") => u32x8,
            _ => u64x4,
        }
    }

    all(
        any(target_arch = "wasm32", target_arch = "wasm64"),
        target_feature = "simd128"
    ) => u64x4,

    all(target_arch = "s390x", target_feature = "vector") => u64x4,

    all(
        any(target_arch = "riscv64", target_arch = "riscv32"),
        target_feature = "zvl128b",
    ) => {
        cfg_select! {
            target_feature = "zve64x" => {
                cfg_select! {
                    target_feature = "zvl4096b" => { Simd<u64, 64> },
                    target_feature = "zvl2048b" => { Simd<u64, 32> },
                    target_feature = "zvl1024b" => u64x16,
                    target_feature = "zvl512b" => u64x8,
                    _ => u64x4,
                }
            }
            target_feature = "zve32x" => {
                cfg_select! {
                    target_feature = "zvl2048b" => { Simd<u32, 64> },
                    target_feature = "zvl1024b" => u32x32,
                    target_feature = "zvl512b" => u32x16,
                    _ => u32x8,
                }
            }
        }
    }

    all(target_arch = "hexagon", target_feature = "hvxv60") => {
        cfg_select! {
            target_feature = "hvx-length128b" => u32x32,
            target_feature = "hvx-length64b" => u32x16,
        }
    }

    all(target_arch = "csky", target_feature = "vdspv1") => u32x8,

    all(
        any(target_arch = "loongarch32", target_arch = "loongarch64"),
        target_feature = "lsx"
    ) => u64x4,
};

pub(super) type Lane = cfg_select! {
    any(
        target_arch = "arm",
        all(
            any(target_arch = "powerpc", target_arch = "powerpc64"),
            target_endian = "big",
            target_feature = "vsx"
        ),
        all(
            any(target_arch = "riscv64", target_arch = "riscv32"),
            not(target_feature = "zve64x")
        ),
        target_arch = "hexagon",
        target_arch = "csky"
    ) => u32,
    _ => u64,
};

pub(super) const REG_BYTES: usize = REG_BITS / 8;
pub(super) const REG_BITS: usize = LANE_BITS * NUM_LANES;
pub(super) const LANE_BITS: usize = Lane::BITS as usize;
pub(super) const NUM_LANES: usize = Reg::LEN;

pub(super) const CANONICAL_LANES: usize = 256 / LANE_BITS;
pub(super) const ROWS_PER_REG: usize = NUM_LANES / CANONICAL_LANES;

#[inline]
pub(super) const fn empty_reg() -> Reg {
    Simd::splat(0)
}

#[inline]
pub(super) fn reg_from_bytes(bytes: [u8; REG_BYTES]) -> Reg {
    Simd::from_be_bytes(Simd::<u8, REG_BYTES>::from_array(bytes))
}

#[inline]
pub(super) fn reg_to_bytes(reg: Reg) -> [u8; REG_BYTES] {
    reg.to_be_bytes().to_array()
}

#[inline]
pub(super) fn edge_cols(reg: Reg) -> Lane {
    (reg[0] & (1 << (LANE_BITS - 1))) | (reg[NUM_LANES - 1] & 1)
}

#[inline]
pub(super) fn is_top_row_active(reg: Reg) -> bool {
    let arr = reg.to_array();
    let mut i = 0;
    while i < CANONICAL_LANES {
        if arr[i] != 0 {
            return true;
        }
        i += 1;
    }
    false
}

#[inline]
pub(super) fn is_bottom_row_active(reg: Reg) -> bool {
    let arr = reg.to_array();
    let mut i = NUM_LANES - CANONICAL_LANES;
    while i < NUM_LANES {
        if arr[i] != 0 {
            return true;
        }
        i += 1;
    }
    false
}

cfg_select! {
    all(target_arch = "s390x", target_feature = "vector-enhancements-2") => {
        use std::arch::s390x::*;
        use std::mem::transmute;

        #[inline]
        fn shld1(a: Reg, b: Reg) -> Reg {
            // SAFETY: `Reg` (`u64x2`) and `vector_unsigned_long_long` are identical 128-bit vector types,
            // and target feature `vector-enhancements-2` is enabled by the enclosing `cfg_select!` guard.
            unsafe {
                transmute(vec_sldb::<vector_unsigned_long_long, 1>(
                    transmute(a),
                    transmute(b),
                ))
            }
        }

        #[inline]
        fn shrd1(a: Reg, b: Reg) -> Reg {
            // SAFETY: `Reg` (`u64x2`) and `vector_unsigned_long_long` are identical 128-bit vector types,
            // and target feature `vector-enhancements-2` is enabled by the enclosing `cfg_select!` guard.
            unsafe {
                transmute(vec_srdb::<vector_unsigned_long_long, 1>(
                    transmute(a),
                    transmute(b),
                ))
            }
        }
    }
    all(
        any(target_arch = "loongarch32", target_arch = "loongarch64"),
        target_feature = "lasx"
    ) => {
        #[cfg(target_arch = "loongarch32")]
        use std::arch::loongarch32::*;
        #[cfg(target_arch = "loongarch64")]
        use std::arch::loongarch64::*;

        #[inline]
        fn shld1(a: Reg, b: Reg) -> Reg {
            let a = a.into();
            let b = b.into();
            // SAFETY: `lasx_xvshuf4i_d`, `lasx_xvpermi_q`, `lasx_xvor_v`, `lasx_xvslli_d`, and
            // `lasx_xvsrli_d` are pure value-to-value SIMD register intrinsics guarded by `target_feature = "lasx"`.
            unsafe {
                let carry_lanes = lasx_xvshuf4i_d::<0x33>(a, lasx_xvpermi_q::<0x03>(a, b));
                lasx_xvor_v(lasx_xvslli_d::<1>(a), lasx_xvsrli_d::<63>(carry_lanes)).into()
            }
        }

        #[inline]
        fn shrd1(a: Reg, b: Reg) -> Reg {
            let a = a.into();
            let b = b.into();
            // SAFETY: `lasx_xvshuf4i_d`, `lasx_xvpermi_q`, `lasx_xvor_v`, `lasx_xvslli_d`, and
            // `lasx_xvsrli_d` are pure value-to-value SIMD register intrinsics guarded by `target_feature = "lasx"`.
            unsafe {
                let carry_lanes = lasx_xvshuf4i_d::<0x99>(b, lasx_xvpermi_q::<0x03>(a, b));
                lasx_xvor_v(lasx_xvsrli_d::<1>(b), lasx_xvslli_d::<63>(carry_lanes)).into()
            }
        }
    }
    _ => {
        #[inline]
        fn shld1(a: Reg, b: Reg) -> Reg {
            let shifted_lanes = a.shift_elements_left::<1>(0);
            let b_shifted = b.shift_elements_right::<{ CANONICAL_LANES - 1 }>(0);

            // Create a mask that is !0 only at the end of each 256-bit boundary (lanes 3, 7, 11...)
            let mut mask_arr = [0; NUM_LANES];
            let mut i = CANONICAL_LANES - 1;
            while i < NUM_LANES {
                mask_arr[i] = !0;
                i += CANONICAL_LANES;
            }
            let mask = Reg::from_array(mask_arr);

            let carry_lanes = (shifted_lanes & !mask) | (b_shifted & mask);
            (a << Simd::splat(1)) | (carry_lanes >> Simd::splat(LANE_BITS as Lane - 1))
        }

        #[inline]
        fn shrd1(a: Reg, b: Reg) -> Reg {
            let shifted_lanes = b.shift_elements_right::<1>(0);
            let a_shifted = a.shift_elements_left::<{ CANONICAL_LANES - 1 }>(0);

            // Create a mask that is !0 only at the start of each 256-bit boundary (lanes 0, 4, 8...)
            let mut mask_arr = [0; NUM_LANES];
            let mut i = 0;
            while i < NUM_LANES {
                mask_arr[i] = !0;
                i += CANONICAL_LANES;
            }
            let mask = Reg::from_array(mask_arr);

            let carry_lanes = (shifted_lanes & !mask) | (a_shifted & mask);
            (b >> Simd::splat(1)) | (carry_lanes << Simd::splat(LANE_BITS as Lane - 1))
        }
    }
}

/// Shifts the internal 256-bit rows "up" (North), pulling in the top row of the South neighbor.
#[inline]
pub(super) fn shift_regs_north(current: Reg, south: Reg) -> Reg {
    if ROWS_PER_REG <= 1 {
        return south;
    }
    // Shift current's bottom row (lanes 4-7) up to the top row (lanes 0-3).
    // `shift_elements_left` moves values towards index 0.
    let up = current.shift_elements_left::<{ CANONICAL_LANES }>(0);
    // Shift south's top row (lanes 0-3) down to the bottom row (lanes 4-7).
    let in_south = south.shift_elements_right::<{ NUM_LANES - CANONICAL_LANES }>(0);
    up | in_south
}

/// Shifts the internal 256-bit rows "down" (South), pulling in the bottom row of the North neighbor.
#[inline]
pub(super) fn shift_regs_south(current: Reg, north: Reg) -> Reg {
    if ROWS_PER_REG <= 1 {
        return north;
    }
    // Shift current's top row (lanes 0-3) down to the bottom row (lanes 4-7).
    let down = current.shift_elements_right::<{ CANONICAL_LANES }>(0);
    // Shift north's bottom row (lanes 4-7) up to the top row (lanes 0-3).
    let in_north = north.shift_elements_left::<{ NUM_LANES - CANONICAL_LANES }>(0);
    down | in_north
}

#[inline]
fn half_add(a: Reg, b: Reg) -> (Reg, Reg) {
    (a ^ b, a & b)
}

#[inline]
fn accumulate((sum, carry0): (Reg, Reg), c: Reg) -> (Reg, Reg) {
    let (sum, carry1) = half_add(sum, c);
    (sum, carry0 | carry1)
}

#[inline]
fn full_add(a: Reg, b: Reg, c: Reg) -> (Reg, Reg) {
    accumulate(half_add(a, b), c)
}

#[inline]
pub(super) fn rule(top: (Reg, Reg), middle: (Reg, Reg), bottom: (Reg, Reg), current: Reg) -> Reg {
    let (hi_parity, hi_carry) = full_add(top.1, middle.1, bottom.1);
    let (lo_parity, lo_carry) = full_add(top.0, middle.0, bottom.0);

    (current | lo_parity) & (hi_parity ^ lo_carry) & !hi_carry
}

#[derive(Clone, Copy)]
pub(super) struct RegDigest {
    pub(super) reg: Reg,
    pub(super) doublet: (Reg, Reg),
    pub(super) triplet: (Reg, Reg),
}

impl RegDigest {
    #[inline]
    pub(super) fn new(reg: Reg, left: Reg, right: Reg) -> Self {
        let doublet = half_add(shrd1(left, reg), shld1(reg, right));
        let triplet = accumulate(doublet, reg);

        Self {
            reg,
            doublet,
            triplet,
        }
    }
}

#[cfg(test)]
mod reg_tests {
    use std::array;

    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    type Bits = [bool; REG_BITS];

    fn bit_iter(a: &u8) -> impl Iterator<Item = bool> {
        (0..8).map(move |i| a & (0x80 >> i) != 0)
    }

    /// Column 0 is the MSB of byte 0, matching `reg_to_bytes` and the parser.
    fn to_bools(reg: Reg) -> Bits {
        let bytes = reg_to_bytes(reg);
        let mut out = [false; REG_BITS];
        for (dst, bit) in out.iter_mut().zip(bytes.iter().flat_map(bit_iter)) {
            *dst = bit;
        }
        out
    }

    fn from_bools(bits: &Bits) -> Reg {
        let mut bytes = [0u8; REG_BYTES];
        for (j, &b) in bits.iter().enumerate() {
            if b {
                bytes[j / 8] |= 0x80 >> (j % 8);
            }
        }
        reg_from_bytes(bytes)
    }

    /// Every lane boundary and its neighbours, as single-bit regs.
    fn edge_regs() -> impl Iterator<Item = Reg> {
        [0, 1, REG_BITS - 2, REG_BITS - 1]
            .into_iter()
            .chain((1..NUM_LANES).flat_map(|lane| [lane * LANE_BITS - 1, lane * LANE_BITS]))
            .map(|j| {
                let mut bits = [false; REG_BITS];
                bits[j] = true;
                from_bools(&bits)
            })
            .chain([empty_reg(), !empty_reg()])
    }

    /// Edge cases followed by `n` seeded random regs.
    fn cases(n: usize, seed: u64) -> impl Iterator<Item = Reg> {
        let mut rng = StdRng::seed_from_u64(seed);
        edge_regs().chain((0..n).map(move |_| reg_from_bytes(rng.random())))
    }

    fn ref_shld1(a: &Bits, b: &Bits) -> Bits {
        array::from_fn(|j| if j + 1 < REG_BITS { a[j + 1] } else { b[0] })
    }

    fn ref_shrd1(a: &Bits, b: &Bits) -> Bits {
        array::from_fn(|j| if j > 0 { b[j - 1] } else { a[REG_BITS - 1] })
    }

    #[test]
    fn shld1_matches_reference() {
        for a in cases(64, 1) {
            for b in cases(64, 2) {
                assert_eq!(to_bools(shld1(a, b)), ref_shld1(&to_bools(a), &to_bools(b)));
            }
        }
    }

    #[test]
    fn shrd1_matches_reference() {
        for a in cases(64, 3) {
            for b in cases(64, 4) {
                assert_eq!(to_bools(shrd1(a, b)), ref_shrd1(&to_bools(a), &to_bools(b)));
            }
        }
    }
}
