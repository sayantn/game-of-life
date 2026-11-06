#![feature(portable_simd)]
#![cfg_attr(
    all(target_arch = "s390x", target_feature = "vector-enhancements-2"),
    feature(stdarch_s390x)
)]
#![cfg_attr(
    all(
        any(target_arch = "loongarch32", target_arch = "loongarch64"),
        target_feature = "lasx"
    ),
    feature(stdarch_loongarch)
)]

pub mod cell;
pub mod chunk;
pub mod infinite_flat_world;
pub mod oldchunk;
mod row;
pub mod toroidalworld;
