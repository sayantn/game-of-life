use crate::{Borrowed, BorrowedMut, Buffer, Orth, Owned};

/// The concrete `Buffer` type associated with the given `Backend`.
pub type BufferOf<B> = <B as Backend>::Buffer;
/// The configuration type required to initialize the backend's buffer.
pub type ConfigOf<B> = <BufferOf<B> as Buffer>::Config;
/// The raw slot index or handle type utilized by the backend's buffer.
pub type HandleOf<B> = <BufferOf<B> as Buffer>::Handle;
/// The `Owned` capability token specific to the backend's buffer.
pub type OwnedOf<B> = Owned<HandleOf<B>>;
/// The `Borrowed` (shared) capability token specific to the backend's buffer.
pub type BorrowedOf<'a, B> = Borrowed<'a, HandleOf<B>>;
/// The `BorrowedMut` (exclusive) capability token specific to the backend's buffer.
pub type BorrowedMutOf<'a, B> = BorrowedMut<'a, HandleOf<B>>;

/// Defines a compute backend (e.g., AVX2 CPU, OpenCL GPU) by pairing a host-side chunk
/// representation, its backing slot storage, and its native activity summary layout.
pub trait Backend: 'static {
    /// Native activity summary produced by [`Buffer::step_batched`].
    ///
    /// Allows each backend to choose its fastest representation (e.g., unpacked booleans on CPU
    /// to avoid bit-packing overhead, vs. a bit-packed `u8` on GPU to minimize PCIe transfer).
    type Activity: Activity;

    /// Backing slot storage where simulation steps are executed.
    type Buffer: Buffer<Activity = Self::Activity>;
}

/// Summary of live-cell presence in a chunk and along its four orthogonal borders after a step.
pub trait Activity: 'static + Copy + Eq {
    /// Pessimistic activity summary marking the chunk and all four edges as active.
    const ALL: Self;

    /// Optimistic activity summary marking the chunk and all four edges as inactive.
    const EMPTY: Self;

    /// Returns `true` if the chunk contains at least one live cell.
    #[must_use]
    fn chunk(self) -> bool;

    /// Returns `true` if the border along `orth` contains at least one live cell.
    #[must_use]
    fn edge(self, orth: Orth) -> bool;
}
