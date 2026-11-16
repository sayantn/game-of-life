use std::ops::Deref;

use crate::traits::{Buffer, Chunk};
use crate::{Backend, BufferOf, ConfigOf, OwnedOf};

/// Free-list slot allocator wrapping a [`Buffer`], recycling freed [`Owned`](crate::Owned)
/// capability tokens in $O(1)$ without reallocating backing device or host memory.
///
/// Dereferences to [`BufferOf<B>`] for direct slot I/O and batched step execution.
#[must_use]
pub struct Arena<B: Backend> {
    buffer: BufferOf<B>,
    free: Vec<OwnedOf<B>>,
}

impl<B: Backend> Arena<B> {
    /// Creates an empty `Arena` with an initialized backend buffer and no pre-allocated slots.
    pub fn new(config: ConfigOf<B>) -> Self {
        Self {
            buffer: BufferOf::<B>::new(config),
            free: Vec::new(),
        }
    }

    /// Allocates a slot from the free list, growing the underlying [`Buffer`] if none are free.
    ///
    /// The returned slot's contents are unspecified and must be overwritten before reading.
    #[must_use]
    pub fn alloc(&mut self) -> OwnedOf<B> {
        self.free.pop().unwrap_or_else(|| {
            self.buffer.grow(1, &mut self.free);
            // SAFETY: `Buffer::grow(1, ...)` guarantees `new_cap >= old_cap + 1`, so at least
            // one newly minted `Owned` handle was pushed onto `self.free`.
            unsafe { self.free.pop().unwrap_unchecked() }
        })
    }

    /// Allocates a slot initialized to all-dead cells (`Chunk::empty()`).
    #[must_use]
    pub fn alloc_zeroed(&mut self) -> OwnedOf<B> {
        self.own(&B::Chunk::empty())
    }

    /// Reclaims `idx` into the free list for future [`alloc`](Self::alloc) calls.
    ///
    /// # Safety
    /// - `idx` must have been minted by `self`'s backing [`Buffer`] (e.g., via [`alloc`](Self::alloc)
    ///   or [`own`](Self::own))
    pub unsafe fn dealloc(&mut self, idx: OwnedOf<B>) {
        self.free.push(idx);
    }

    /// Allocates a slot and initializes it with the contents of `chunk`.
    #[must_use]
    pub fn own(&mut self, chunk: &B::Chunk) -> OwnedOf<B> {
        let mut idx = self.alloc();
        // SAFETY: `idx` was minted by `self.buffer` via `self.alloc()`.
        unsafe { self.buffer.store(&mut idx, chunk) };
        idx
    }

    /// Creates a new, empty `Arena` sharing the same backend context/configuration as `self`
    /// with an empty free list.
    pub fn empty_clone(&self) -> Self {
        Self {
            buffer: self.buffer.empty_clone(),
            free: Vec::new(),
        }
    }
}

impl<B: Backend> Deref for Arena<B> {
    type Target = BufferOf<B>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
