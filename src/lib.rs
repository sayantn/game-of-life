#![feature(portable_simd, vec_push_within_capacity, sync_unsafe_cell)]
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
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::wildcard_imports,
    clippy::doc_markdown
)]
#![deny(clippy::missing_safety_doc)]

pub mod arena;
pub mod backend;
pub mod helpers;
mod infinite_flat_world;
mod primitives;
mod toroidal_world;
mod traits;

pub use arena::Arena;
pub use backend::DefaultBackend;
pub use infinite_flat_world::InfiniteFlatWorld;
pub use primitives::*;
pub use toroidal_world::ToroidalWorld;
pub use traits::*;

pub type DefaultChunk = ChunkOf<DefaultBackend>;

#[allow(unused)]
#[cfg(test)]
mod naive;
