#![feature(portable_simd, vec_push_within_capacity)]
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
#![allow(
    clippy::must_use_candidate,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::wildcard_imports,
    clippy::return_self_not_must_use
)]

mod arena;
pub mod chunk;
// pub mod infinite_flat_world;
pub mod infinite_flat_world_2;
mod parse;
mod row;
pub mod toroidal_world;

#[allow(unused)]
#[cfg(test)]
mod naive;
