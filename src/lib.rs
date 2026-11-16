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
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::wildcard_imports,
    clippy::chunks_exact_to_as_chunks
)]
#![warn(
    missing_debug_implementations,
    missing_copy_implementations,
    clippy::ptr_as_ptr,
    clippy::trivially_copy_pass_by_ref,
    clippy::missing_const_for_fn,
    // missing_docs
)]
#![deny(
    clippy::missing_safety_doc,
    clippy::missing_inline_in_public_items,
    clippy::undocumented_unsafe_blocks,
    clippy::unnecessary_safety_comment,
    clippy::unnecessary_safety_doc,
    unsafe_op_in_unsafe_fn
)]

mod arena;
pub mod backend;
mod buffer;
mod chunk;
mod helpers;
mod infinite_flat_world;
mod primitives;
mod toroidal_world;
mod traits;

pub use arena::Arena;
pub use backend::DefaultBackend;
pub use buffer::*;
pub use chunk::*;
pub use infinite_flat_world::InfiniteFlatWorld;
pub use primitives::*;
pub use toroidal_world::ToroidalWorld;
pub use traits::*;

#[allow(unused)]
#[cfg(test)]
mod naive;
