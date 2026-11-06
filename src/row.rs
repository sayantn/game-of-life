use std::simd::num::SimdUint;
use std::{mem, simd::*};

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

pub(crate) const ROW_BYTES: usize = mem::size_of::<Row>();

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
        pub(crate) fn shld1(a: Row, b: Row) -> Row {
            a.funnel_shl(b, 1)
        }

        #[inline]
        pub(crate) fn shrd1(a: Row, b: Row) -> Row {
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
