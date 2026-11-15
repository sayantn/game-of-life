pub(crate) type Row = cfg_select! {
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ) => {
        cfg_select! {
            target_feature = "avx512f" => u64x8,
            target_feature = "avx2" => u64x4,
            _ => u64x2,
        }
    }

    all(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        target_feature = "neon"
    ) => u64x2,

    all(
        target_arch = "arm",
        any(target_feature = "neon", target_feature = "mve")
    ) => u32x4,

    all(
        any(target_arch = "powerpc", target_arch = "powerpc64"),
        target_feature = "altivec"
    ) => {
        cfg_select! {
            all(target_endian = "big", target_feature = "vsx") => u32x4,
            _ => u64x2,
        }
    }

    all(
        any(target_arch = "wasm32", target_arch = "wasm64"),
        target_feature = "simd128"
    ) => u64x2,

    all(target_arch = "s390x", target_feature = "vector") => u64x2,

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
                    target_feature = "zvl256b" => u64x4,
                    _ => u64x2,
                }
            }
            target_feature = "zve32x" => {
                cfg_select! {
                    target_feature = "zvl2048b" => { Simd<u32, 64> },
                    target_feature = "zvl1024b" => u32x32,
                    target_feature = "zvl512b" => u32x16,
                    target_feature = "zvl256b" => u32x8,
                    _ => u32x4,
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

    all(target_arch = "csky", target_feature = "vdspv1") => u32x4,

    all(
        any(target_arch = "loongarch32", target_arch = "loongarch64"),
        target_feature = "lsx"
    ) => {
        cfg_select! {
            target_feature = "lasx" => u64x4,
            _ => u64x2,
        }
    }

    _ => u64,
};

pub(crate) const ROW_BYTES: usize = ROW_BITS / 8;
pub(crate) const ROW_BITS: usize = LANE_BITS * NUM_LANES;

#[allow(unused)]
pub(crate) const LANE_BITS: usize = Lane::BITS as usize;

cfg_select! {
    any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        ),
        all(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            target_feature = "neon"
        ),
        all(
            target_arch = "arm",
            any(target_feature = "neon", target_feature = "mve")
        ),
        all(
            any(target_arch = "powerpc", target_arch = "powerpc64"),
            target_feature = "altivec"
        ),
        all(
            any(target_arch = "wasm32", target_arch = "wasm64"),
            target_feature = "simd128"
        ),
        all(target_arch = "s390x", target_feature = "vector"),
        all(
            any(target_arch = "riscv64", target_arch = "riscv32"),
            target_feature = "zvl128b",
        ),
        all(target_arch = "hexagon", target_feature = "hvxv60"),
        all(target_arch = "csky", target_feature = "vdspv1"),
        all(
            any(target_arch = "loongarch32", target_arch = "loongarch64"),
            target_feature = "lsx"
        )
    ) => {
        use std::simd::{num::SimdUint, *};

        pub(crate) const NUM_LANES: usize = Row::LEN;

        #[inline]
        pub(crate) fn empty_row() -> Row {
            Simd::splat(0)
        }

        #[inline]
        pub(crate) fn row_from_bytes(bytes: [u8; ROW_BYTES]) -> Row {
            Simd::from_be_bytes(Simd::<u8, ROW_BYTES>::from_array(bytes))
        }

        #[inline]
        pub(crate) fn row_to_bytes(row: Row) -> [u8; ROW_BYTES] {
            row.to_be_bytes().to_array()
        }

        #[inline]
        pub(crate) fn row_num_alive(row: Row) -> usize {
            row.count_ones().reduce_sum() as _
        }

        #[inline]
        pub(crate) fn edge_cols(row: Row) -> Lane {
            (row[0] & (1 << (LANE_BITS - 1))) | (row[NUM_LANES - 1] & 1)
        }

        pub(crate) type Lane = cfg_select! {
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

        cfg_select! {
            all(target_arch = "s390x", target_feature = "vector-enhancements-2") => {
                use std::arch::s390x::*;
                use std::mem::transmute;

                #[inline]
                fn shld1(a: Row, b: Row) -> Row {
                    unsafe {
                        transmute(vec_sldb::<vector_unsigned_long_long, 1>(
                            transmute(a),
                            transmute(b),
                        ))
                    }
                }

                #[inline]
                fn shrd1(a: Row, b: Row) -> Row {
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
                fn shld1(a: Row, b: Row) -> Row {
                    let a = a.into();
                    let b = b.into();
                    unsafe {
                        let carry_lanes = lasx_xvshuf4i_d::<0x33>(a, lasx_xvpermi_q::<0x03>(a, b));
                        lasx_xvor_v(lasx_xvslli_d::<1>(a), lasx_xvsrli_d::<63>(carry_lanes)).into()
                    }
                }

                #[inline]
                fn shrd1(a: Row, b: Row) -> Row {
                    let a = a.into();
                    let b = b.into();
                    unsafe {
                        let carry_lanes = lasx_xvshuf4i_d::<0x99>(b, lasx_xvpermi_q::<0x03>(a, b));
                        lasx_xvor_v(lasx_xvsrli_d::<1>(b), lasx_xvslli_d::<63>(carry_lanes)).into()
                    }
                }
            }
            _ => {
                #[inline]
                fn shld1(a: Row, b: Row) -> Row {
                    let carry_lanes = a.shift_elements_left::<1>(0)
                        | b.shift_elements_right::<{ Row::LEN - 1 }>(0);
                    (a << Simd::splat(1)) | (carry_lanes >> Simd::splat(63))
                }

                #[inline]
                fn shrd1(a: Row, b: Row) -> Row {
                    let carry_lanes = b.shift_elements_right::<1>(0)
                        | a.shift_elements_left::<{ Row::LEN - 1 }>(0);
                    (b >> Simd::splat(1)) | (carry_lanes << Simd::splat(63))
                }
            }
        }
    }
    _ => {
        pub(crate) const NUM_LANES: usize = 1;

        pub(crate) type Lane = u64;

        #[inline]
        pub(crate) fn empty_row() -> Row {
            0
        }

        #[inline]
        pub(crate) fn row_from_bytes(bytes: [u8; ROW_BYTES]) -> Row {
            Row::from_be_bytes(bytes)
        }

        #[inline]
        pub(crate) fn row_to_bytes(row: Row) -> [u8; ROW_BYTES] {
            row.to_be_bytes()
        }

        #[inline]
        pub(crate) fn row_num_alive(row: Row) -> usize {
            row.count_ones() as _
        }

        #[inline]
        pub(crate) fn edge_cols(row: Row) -> Lane {
            (row & (1 << (LANE_BITS - 1))) | (row & 1)
        }

        #[inline]
        fn shld1(a: Row, b: Row) -> Row {
            a.funnel_shl(b, 1)
        }

        #[inline]
        fn shrd1(a: Row, b: Row) -> Row {
            a.funnel_shr(b, 1)
        }
    }
}

#[inline]
fn half_add(a: Row, b: Row) -> (Row, Row) {
    (a ^ b, a & b)
}

#[inline]
fn accumulate((sum, carry0): (Row, Row), c: Row) -> (Row, Row) {
    let (sum, carry1) = half_add(sum, c);
    (sum, carry0 | carry1)
}

#[inline]
fn full_add(a: Row, b: Row, c: Row) -> (Row, Row) {
    accumulate(half_add(a, b), c)
}

#[inline]
pub(crate) fn rule(top: (Row, Row), middle: (Row, Row), bottom: (Row, Row), current: Row) -> Row {
    let (hi_parity, hi_carry) = full_add(top.1, middle.1, bottom.1);
    let (lo_parity, lo_carry) = full_add(top.0, middle.0, bottom.0);

    (current | lo_parity) & (hi_parity ^ lo_carry) & !hi_carry
}

#[derive(Clone, Copy)]
pub(crate) struct RowDigest {
    pub(crate) row: Row,
    pub(crate) doublet: (Row, Row),
    pub(crate) triplet: (Row, Row),
}

impl RowDigest {
    #[inline]
    pub(crate) fn new(row: Row, left: Row, right: Row) -> Self {
        let doublet = half_add(shrd1(left, row), shld1(row, right));
        let triplet = accumulate(doublet, row);

        Self {
            row,
            doublet,
            triplet,
        }
    }
}

#[cfg(test)]
mod row_tests {
    use std::array;

    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    type Bits = [bool; ROW_BITS];

    fn bit_iter(a: &u8) -> impl Iterator<Item = bool> {
        (0..8).map(move |i| a & (0x80 >> i) != 0)
    }

    /// Column 0 is the MSB of byte 0, matching `row_to_bytes` and the parser.
    fn to_bools(row: Row) -> Bits {
        let bytes = row_to_bytes(row);
        let mut out = [false; ROW_BITS];
        for (dst, bit) in out.iter_mut().zip(bytes.iter().flat_map(bit_iter)) {
            *dst = bit;
        }
        out
    }

    fn from_bools(bits: &Bits) -> Row {
        let mut bytes = [0u8; ROW_BYTES];
        for (j, &b) in bits.iter().enumerate() {
            if b {
                bytes[j / 8] |= 0x80 >> (j % 8);
            }
        }
        row_from_bytes(bytes)
    }

    /// Every lane boundary and its neighbours, as single-bit rows.
    fn edge_rows() -> impl Iterator<Item = Row> {
        [0, 1, ROW_BITS - 2, ROW_BITS - 1]
            .into_iter()
            .chain((1..NUM_LANES).flat_map(|lane| [lane * LANE_BITS - 1, lane * LANE_BITS]))
            .map(|j| {
                let mut bits = [false; ROW_BITS];
                bits[j] = true;
                from_bools(&bits)
            })
            .chain([empty_row(), !empty_row()])
    }

    /// Edge cases followed by `n` seeded random rows.
    fn cases(n: usize, seed: u64) -> impl Iterator<Item = Row> {
        let mut rng = StdRng::seed_from_u64(seed);
        edge_rows().chain((0..n).map(move |_| row_from_bytes(rng.random())))
    }

    fn ref_shld1(a: &Bits, b: &Bits) -> Bits {
        array::from_fn(|j| if j + 1 < ROW_BITS { a[j + 1] } else { b[0] })
    }

    fn ref_shrd1(a: &Bits, b: &Bits) -> Bits {
        array::from_fn(|j| if j > 0 { b[j - 1] } else { a[ROW_BITS - 1] })
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
