use core::marker::PhantomData;

use crate::helpers::recycle_vec;
use crate::{Activity, Chunk, NeighborData};

/// Affine ownership token for a uniquely owned slot in a `Buffer`
/// whose raw slot descriptor is `D`.
#[derive(Debug)]
#[repr(transparent)]
pub struct Owned<D: Copy> {
    raw: D,
}

impl<D: Copy> Owned<D> {
    /// Mints an owned capability token from a raw backend descriptor.
    ///
    /// # Safety
    /// The caller must ensure `raw` refers to a valid, uniquely owned slot
    /// in the target buffer and is not aliased by any other `Owned<D>`.
    #[inline]
    pub const unsafe fn from_raw(raw: D) -> Self {
        Self { raw }
    }

    /// Borrows this owned slot, yielding a shared read-only capability.
    #[inline]
    pub const fn borrow(&self) -> Borrowed<'_, D> {
        Borrowed {
            raw: self.raw,
            _lifetime: PhantomData,
        }
    }

    /// Mutably borrows this owned slot, yielding an exclusive write capability.
    #[inline]
    pub const fn borrow_mut(&mut self) -> BorrowedMut<'_, D> {
        BorrowedMut {
            raw: self.raw,
            _lifetime: PhantomData,
        }
    }

    /// Extracts the raw backend descriptor without dropping the token.
    #[inline]
    pub const fn raw(&self) -> D {
        self.raw
    }

    /// Consumes the token and returns the raw backend descriptor.
    #[inline]
    pub const fn into_raw(self) -> D {
        self.raw
    }
}

/// Shared, covariant, `Copy` capability handle to a read-only buffer slot.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Borrowed<'a, D: Copy> {
    raw: D,
    _lifetime: PhantomData<fn() -> &'a ()>,
}

impl<D: Copy> Borrowed<'_, D> {
    /// Extracts the raw backend descriptor for this read-only handle.
    #[inline]
    pub const fn raw(self) -> D {
        self.raw
    }
}

/// Exclusive, invariant, non-`Copy` capability handle to a writable buffer slot.
#[derive(Debug)]
#[repr(transparent)]
pub struct BorrowedMut<'a, D: Copy> {
    raw: D,
    _lifetime: PhantomData<fn(&'a ()) -> &'a ()>,
}

impl<D: Copy> BorrowedMut<'_, D> {
    /// Extracts the raw backend descriptor for this exclusive handle.
    #[inline]
    pub const fn raw(&self) -> D {
        self.raw
    }

    /// Temporarily yields a shared read-only capability from this exclusive capability.
    #[inline]
    pub const fn borrow(&self) -> Borrowed<'_, D> {
        Borrowed {
            raw: self.raw,
            _lifetime: PhantomData,
        }
    }

    /// Re-borrows this exclusive capability with a shorter lifetime.
    #[inline]
    pub const fn borrow_mut(&mut self) -> BorrowedMut<'_, D> {
        BorrowedMut {
            raw: self.raw,
            _lifetime: PhantomData,
        }
    }
}

impl<'a, D: Copy> From<&'a mut Owned<D>> for BorrowedMut<'a, D> {
    #[inline]
    fn from(owned: &'a mut Owned<D>) -> Self {
        owned.borrow_mut()
    }
}

/// Single-generation transition job for an active chunk with an allocated center.
/// Exact layout equivalence with `[D; 10]`.
#[repr(C)]
#[derive(Debug)]
pub struct ComputeJob<'a, D: Copy> {
    /// Read-only handles to the 8 surrounding neighbor chunks.
    pub neighbors: NeighborData<Borrowed<'a, D>>,
    /// Read-only handle to the center chunk representing the current state.
    pub center: Borrowed<'a, D>,
    /// Exclusive handle to the destination slot for the next generation.
    pub dst: BorrowedMut<'a, D>,
}

/// Single-generation transition job for an unallocated (implicitly zero) frontier chunk,
/// where only live cells on the 8 neighbors' borders can propagate inward.
/// Exact layout equivalence with `[D; 9]`.
#[repr(C)]
#[derive(Debug)]
pub struct VacantComputeJob<'a, D: Copy> {
    /// Read-only handles to the 8 surrounding neighbor chunks.
    pub neighbors: NeighborData<Borrowed<'a, D>>,
    /// Exclusive handle to the destination slot for the next generation.
    pub dst: BorrowedMut<'a, D>,
}

/// Low-level, fixed-slot-size backing storage for chunk data (e.g., a CPU `Vec<UnsafeCell<Chunk>>`
/// or an OpenCL `cl_mem` buffer), decoupling physical memory management from `Arena<B>`'s
/// free-list allocator.
///
/// # Safety
///
/// Implementations of `Buffer` must uphold the following invariants so callers can rely on
/// type-level capability guarantees without runtime bounds checks or aliasing checks:
///
/// 1. **Handle Exclusivity & Uniqueness:**
///    - [`Owned`](crate::Owned) represents exclusive capability to a single physical slot in `self`.
///    - [`new`](Self::new) must create a buffer with `capacity() == 0` and zero minted handles.
///    - [`grow`](Self::grow) and [`grow_exact`](Self::grow_exact) must increase [`capacity`](Self::capacity)
///      by at least `additional` slots (`new_cap >= old_cap + additional`) and mint fresh `Owned` tokens strictly
///      for the newly added slots `old_cap..new_cap`, never duplicating any previously minted handle.
///
/// 2. **Stability Across Reallocation:**
///    - Calling `grow` or `grow_exact` must preserve the validity, identity, and contents of all existing slots
///      `0..old_cap`.
///    - Any `Owned` handle minted prior to reallocation must continue to refer to the exact same logical
///      slot after reallocation completes.
///    - Any capability token must represent a logical index or stable ID, rather than a raw pointer
///      that would be invalidated by slab reallocation.
///
/// 3. **Interior Mutability:**
///    - [`store`](Self::store), [`copy`](Self::copy), and [`step_batched`](Self::step_batched)
///      take `&self` and mutate the backing storage via interior mutability (or device command
///      queues). The implementation must guarantee that concurrent writes through disjoint
///      `BorrowedMut` capabilities do not cause data races.
pub unsafe trait Buffer: 'static + Sized {
    /// Backend-specific configuration or context required to initialize the buffer
    /// (e.g., `()` for CPU memory, or an OpenCL context/queue handle for GPU memory).
    type Config: Clone;

    /// Native activity summary produced by [`Buffer::step_batched`].
    ///
    /// Allows each backend to choose its fastest representation (e.g., unpacked booleans on CPU
    /// to avoid bit-packing overhead, vs. a bit-packed `u8` on GPU to minimize PCIe transfer).
    type Activity: Activity;

    /// The raw internal handle type used by the backend to index slots.
    type Handle: Copy;

    /// Creates an empty buffer with `capacity() == 0` and performs no slot allocations.
    #[must_use]
    fn new(config: Self::Config) -> Self;

    /// Returns a cloned copy of the configuration/context used to construct this buffer,
    /// allowing callers to spawn new, independent buffers sharing the same backend resources.
    #[must_use]
    fn config(&self) -> &Self::Config;

    /// Returns the total number of slots currently allocated in this buffer.
    #[must_use]
    fn capacity(&self) -> usize;

    /// Grows the buffer's capacity by at least `additional` slots (`new_cap >= old_cap + additional`),
    /// preserving all existing slots and appending all `new_cap - old_cap` newly minted
    /// [`Owned`](crate::Owned) tokens to `out`.
    ///
    /// Like [`Vec::reserve`], the implementation may allocate more than `additional` slots
    /// to amortize reallocation costs or align to backend-specific page/workgroup boundaries.
    /// If `additional == 0`, this may be a no-op.
    ///
    /// # Panics
    /// Panics if the new capacity overflows `usize` or fails to allocate.
    #[inline]
    fn grow(&mut self, additional: usize, out: &mut Vec<Owned<Self::Handle>>) {
        self.grow_exact(additional, out);
    }

    /// Grows the buffer's capacity by `additional` slots (`new_cap >= old_cap + additional`),
    /// avoiding exponential over-allocation while still allowing backends to round up to
    /// required alignment or workgroup boundaries.
    ///
    /// Appends all `new_cap - old_cap` newly minted [`Owned`](crate::Owned) tokens to `out`.
    /// If `additional == 0`, this is a no-op.
    ///
    /// # Panics
    /// Panics if the new capacity overflows `usize` or fails to allocate.
    fn grow_exact(&mut self, additional: usize, out: &mut Vec<Owned<Self::Handle>>);

    /// Reads the chunk at `id` and returns an owned host copy.
    ///
    /// # Safety
    /// - `id` must have been minted by `self`.
    #[inline]
    #[must_use]
    unsafe fn load(&self, id: Borrowed<'_, Self::Handle>) -> Chunk {
        let mut chunk = Chunk::empty();
        // SAFETY: Caller guarantees `id` was minted by `self`.
        unsafe { self.load_to(id, &mut chunk) };
        chunk
    }

    /// Reads the chunk at `id` into `dst`.
    ///
    /// # Safety
    /// - `id` must have been minted by `self`.
    unsafe fn load_to(&self, id: Borrowed<'_, Self::Handle>, dst: &mut Chunk);

    /// Overwrites the slot represented by `dst` with `src`.
    ///
    /// # Safety
    /// - `dst` must have been minted by `self`.
    unsafe fn store(&self, dst: BorrowedMut<'_, Self::Handle>, src: &Chunk);

    /// Copies the chunk at `src` (from `from`) into the slot `dst` (in `self`).
    ///
    /// # Safety
    /// - `dst` must have been minted by `self`.
    /// - `src` must have been minted by `from`.
    unsafe fn copy(
        &self,
        dst: BorrowedMut<'_, Self::Handle>,
        from: &Self,
        src: Borrowed<'_, Self::Handle>,
    );

    /// Executes a batch of allocated and vacant-frontier chunk transitions.
    ///
    /// When `activities` is `Some(out)`, `out` is cleared and populated with
    /// `jobs.compute.len()` entries followed by `jobs.vacant_compute.len()` entries in order.
    ///
    /// # Safety
    /// - Every handle in `jobs.compute` and `jobs.vacant_compute` must have been allocated by `self`.
    unsafe fn step_batched(
        &self,
        jobs: &mut TickBatch<'_, Self::Handle>,
        activities: Option<&mut Vec<Self::Activity>>,
    );
}

/// Persistent, `'static` heap allocations reused across ticks via `recycle_vec`.
/// Stores no lifetimes and no `Owned` handles—zero self-reference!
#[must_use]
#[derive(Debug)]
pub struct StepScratch<D: Copy> {
    compute: Vec<ComputeJob<'static, D>>,
    vacant_compute: Vec<VacantComputeJob<'static, D>>,
}

impl<D: Copy> Default for StepScratch<D> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Copy> StepScratch<D> {
    /// Creates a new, empty scratch buffer for chunk transitions.
    #[inline]
    pub const fn new() -> Self {
        Self {
            compute: Vec::new(),
            vacant_compute: Vec::new(),
        }
    }

    /// Borrows the underlying heap allocations with fresh lifetime `'a` for one tick.
    #[inline]
    pub fn start_tick<'a>(&mut self) -> TickBatch<'a, D> {
        let compute = core::mem::take(&mut self.compute);
        let vacant_compute = core::mem::take(&mut self.vacant_compute);
        TickBatch {
            compute: recycle_vec(compute),
            vacant_compute: recycle_vec(vacant_compute),
        }
    }

    /// Reclaims the cleared heap allocations at the end of the tick.
    #[inline]
    pub fn finish_tick(&mut self, batch: TickBatch<'_, D>) {
        self.compute = recycle_vec(batch.compute);
        self.vacant_compute = recycle_vec(batch.vacant_compute);
    }
}

/// Ephemeral per-tick job builder borrowing `Owned<D>` handles for lifetime `'a`.
#[derive(Debug)]
pub struct TickBatch<'a, D: Copy> {
    /// Batch of standard transition jobs for fully allocated chunks.
    pub compute: Vec<ComputeJob<'a, D>>,
    /// Batch of transition jobs for border chunks propagating inward.
    pub vacant_compute: Vec<VacantComputeJob<'a, D>>,
}
