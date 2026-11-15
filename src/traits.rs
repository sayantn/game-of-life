use std::fmt::Debug;
use std::str::FromStr;

use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use crate::primitives::{NeighborData, Orth};

/// Defines a compute backend (e.g., AVX2 CPU, OpenCL GPU) by pairing a host-side chunk
/// representation, its backing slot storage, and its native activity summary layout.
pub trait Backend: 'static {
    /// Host-side chunk representation used for serialization, initialization, and debugging.
    type Chunk: Chunk;

    /// Native activity summary produced by [`Buffer::compute_batched`].
    ///
    /// Allows each backend to choose its fastest representation (e.g., unpacked booleans on CPU
    /// to avoid bit-packing overhead, vs. a bit-packed `u8` on GPU to minimize PCIe transfer).
    type Activity: Activity;

    /// Backing slot storage where simulation steps are executed.
    type Buffer: Buffer<Chunk = Self::Chunk, Activity = Self::Activity>;
}

/// Summary of live-cell presence in a chunk and along its four orthogonal borders after a step.
pub trait Activity: Copy + Debug + 'static {
    /// Pessimistic activity summary marking the chunk and all four edges as active.
    const ALL: Self;

    /// Optimistic activity summary marking the chunk and all four edges as inactive.
    const EMPTY: Self;

    /// Returns `true` if the chunk contains at least one live cell.
    fn chunk(self) -> bool;

    /// Returns `true` if the border along `orth` contains at least one live cell.
    fn edge(self, orth: Orth) -> bool;
}

/// Host-side representation of a square `SIZE × SIZE` grid of Game of Life cells.
///
/// Used as the interchange format for initializing worlds, inspecting simulation state,
/// and cross-verifying results between different [`Backend`] implementations.
pub trait Chunk: Sized + Clone + Debug + FromStr + From<Self::Bytes> + 'static {
    /// Side length of the square chunk in cells (e.g., `256` for a 256×256 bitboard).
    const SIZE: usize;

    /// Exact number of bytes required to densely pack `SIZE × SIZE` cells (1 bit per cell).
    const BYTES: usize = Self::SIZE.strict_mul(Self::SIZE).div_ceil(8);

    /// Fixed-size byte container used to serialize and deserialize this chunk.
    ///
    /// Typically a nested array like `[[u8; SIZE / 8]; SIZE]` or a flat `[u8; BYTES]`.
    type Bytes: Clone + AsRef<[u8]> + AsMut<[u8]> + From<Self>;

    fn random(rng: &mut impl Rng) -> Self
    where
        StandardUniform: Distribution<Self::Bytes>,
    {
        Self::from_bytes(&rng.random())
    }

    fn from_bytes(bytes: &Self::Bytes) -> Self;

    fn to_bytes(&self) -> Self::Bytes;

    /// Creates a chunk with all cells dead (zeroed).
    fn empty() -> Self;

    /// Returns the total number of live cells in this chunk.
    fn num_alive(&self) -> usize;

    /// Returns `true` if every cell in this chunk is dead.
    fn is_empty(&self) -> bool;
}

/// Move-only capability token representing exclusive ownership of an allocated chunk slot
/// inside a [`Buffer`].
///
/// Dropping an `Owned` token has no side effects on its own; slots are reclaimed either via
/// an [`Arena`](crate::Arena) or when the parent [`Buffer`] is dropped.
pub trait Owned: 'static {
    /// Lightweight, `Copy` read-only handle tied to the lifetime of a shared borrow (`&'a Self`).
    type Id<'a>: Copy;

    /// Borrows this slot as a read-only [`Id`](Self::Id) for the duration of `'a`.
    ///
    /// While the returned `Id<'a>` is live, Rust's borrow checker prevents mutating
    /// (`&mut Self`) or consuming (`self`) the underlying slot.
    fn id<'a>(&'a self) -> Self::Id<'a>;
}

/// Low-level, fixed-slot-size backing storage for chunk data (e.g., a CPU `Vec<UnsafeCell<Chunk>>`
/// or an OpenCL `cl_mem` buffer), decoupling physical memory management from `Arena<B>`'s
/// free-list allocator.
///
/// # Safety (Implementor Contract)
/// Implementations of `Buffer` must uphold the following invariants so callers can rely on
/// type-level capability guarantees without runtime bounds checks or aliasing checks:
///
/// 1. **Handle Exclusivity & Uniqueness:**
///    - [`Owned`](Self::Owned) represents exclusive write capability to a single physical slot in `self`.
///    - [`Id<'a>`](Self::Id) represents shared read-only capability to that slot for the borrow `'a`.
///    - [`new`](Self::new) must create a buffer with `capacity() == 0` and zero minted handles.
///    - [`grow`](Self::grow) must increase [`capacity`](Self::capacity) by at least `additional`
///      slots (`new_cap >= old_cap + additional`) and mint fresh `Owned` tokens strictly for the
///      newly added slots `old_cap..new_cap`, never duplicating any previously minted `Owned` handle.
///
/// 2. **Stability Across Reallocation ([`grow`](Self::grow)):**
///    - Calling `grow` must preserve the validity, identity, and contents of all existing slots
///      `0..old_cap`.
///    - Any `Owned` handle minted prior to `grow` must continue to refer to the exact same logical
///      slot after `grow` completes.
///    - Any `Id<'a>` created from an `Owned` handle must remain valid across `grow` (meaning
///      `Id<'a>` must be a logical handle/index rather than a raw pointer invalidated by slab
///      reallocation).
///
/// 3. **Handle Correspondence ([`id`](Self::id)):**
///    - `id(&self, owned)` must return an `Id<'a>` referring to the exact same slot as `owned`,
///      without mutating the slot's contents.
///
/// 4. **Interior Mutability & Thread Safety:**
///    - [`store`](Self::store), [`copy`](Self::copy), and [`compute_batched`](Self::compute_batched)
///      take `&self` and mutate the backing storage via interior mutability (or device command
///      queues). Implementations must ensure disjoint slots can be safely read/written
///      concurrently when `Self: Sync`.
pub unsafe trait Buffer: Sized + 'static {
    /// Backend-specific configuration or context required to initialize the buffer
    /// (e.g., `()` for CPU memory, or an OpenCL context/queue handle for GPU memory).
    type Config;

    /// The host-side [`ChunkData`] format associated with this buffer.
    type Chunk: Chunk;

    /// Native activity summary produced by [`Buffer::compute_batched`].
    ///
    /// Allows each backend to choose its fastest representation (e.g., unpacked booleans on CPU
    /// to avoid bit-packing overhead, vs. a bit-packed `u8` on GPU to minimize PCIe transfer).
    type Activity: Activity;

    /// Exclusive ownership token for a single allocated slot in this buffer.
    ///
    /// Must not implement `Clone` or `Copy` in safe code, ensuring each slot has at most one
    /// mutable owner at any time.
    type Owned: for<'a> Owned<Id<'a> = Self::Id<'a>>;

    /// Lightweight, `Copy` read-only capability token borrowing a slot for lifetime `'a`.
    type Id<'a>: Copy;

    /// Creates an empty buffer with `capacity() == 0` and performs no slot allocations.
    fn new(config: Self::Config) -> Self;

    /// Creates a new, empty buffer configured with the same backend context/parameters
    /// as `self`, without copying any existing chunks or minting any `Owned` tokens.
    fn empty_clone(&self) -> Self;

    /// Returns the total number of slots currently allocated in this buffer.
    fn capacity(&self) -> usize;

    /// Grows the buffer's capacity to hold at least `self.capacity() + additional` slots,
    /// preserving all existing slots and appending all `new_cap - old_cap` newly minted
    /// [`Owned`](Self::Owned) tokens to `out`.
    ///
    /// Like [`Vec::reserve`], the implementation may allocate more than `additional` slots
    /// to amortize reallocation costs or align to backend-specific page/workgroup boundaries.
    /// If `additional == 0`, this may be a no-op.
    ///
    /// # Panics
    /// Panics if the new capacity overflows `usize` or fails to allocate.
    fn grow(&mut self, additional: usize, out: &mut Vec<Self::Owned>);

    /// Reads the chunk at `id` and returns an owned host copy.
    ///
    /// # Safety
    /// - `id` must have been minted by `self`.
    unsafe fn load(&self, id: Self::Id<'_>) -> Self::Chunk {
        let mut chunk = Self::Chunk::empty();
        unsafe { self.load_to(id, &mut chunk) };
        chunk
    }

    /// Reads the chunk at `id` into `dst`.
    ///
    /// # Safety
    /// - `id` must have been minted by `self`.
    unsafe fn load_to(&self, id: Self::Id<'_>, dst: &mut Self::Chunk);

    /// Overwrites the slot represented by `dst` with `src`.
    ///
    /// # Safety
    /// - `dst` must have been minted by `self`.
    unsafe fn store(&self, dst: &mut Self::Owned, src: &Self::Chunk);

    /// Overwrites each slot in `dsts` with the corresponding chunk in `srcs`.
    ///
    /// Backends with high per-call transfer overhead (such as GPUs) should override this
    /// to coalesce host-to-device uploads into a single command or staging buffer transfer.
    ///
    /// # Panics
    /// Panics if `dsts.len() != srcs.len()`.
    ///
    /// # Safety
    /// Every `Owned` handle in `dsts` must have been minted by `self`.
    unsafe fn store_batched(&self, dsts: &mut [Self::Owned], srcs: &[Self::Chunk]) {
        assert_eq!(dsts.len(), srcs.len());
        for (dst, src) in dsts.iter_mut().zip(srcs) {
            // SAFETY: Caller guarantees every `dst` in `dsts` was minted by `self`.
            unsafe { self.store(dst, src) };
        }
    }

    /// Folds an arbitrary host function across all chunks in `ids`.
    ///
    /// On host backends, this inspects chunks directly in-place without copying.
    /// On device backends, this streams chunks from device memory into a reusable scratch buffer.
    ///
    /// # Safety
    /// Every `Id<'_>` in `ids` must have been minted by `self`.
    unsafe fn num_alive_batched(&self, ids: &[Self::Id<'_>]) -> usize {
        ids.iter()
            .map(|&id| unsafe { self.load(id).num_alive() })
            .sum()
    }

    /// Copies the chunk at `src` (from `from`) into the slot `dst` (in `self`).
    ///
    /// # Safety
    /// - `dst` must have been minted by `self`.
    /// - `src` must have been minted by `from`.
    unsafe fn copy(&self, dst: &mut Self::Owned, from: &Self, src: Self::Id<'_>);

    /// Executes a batched Game of Life step across all alive and dead chunks in `batch`, writing
    /// resulting states into `alive_dsts` and `dead_dsts` and pushing [`Activity`] summaries to `out`.
    ///
    /// # Panics
    /// Panics if `alive_dsts.len()` does not match `batch.alive_len()`, or if
    /// `dead_dsts.len()` does not match `batch.dead_len()`.
    ///
    /// # Safety
    /// - Every `Id<'_>` in `batch` and every `Owned` handle in `alive_dsts` and `dead_dsts` must
    ///   have been minted by `self`.
    unsafe fn compute_batched(
        &self,
        batch: &Batch<'_, Self>,
        alive_dsts: &mut [Self::Owned],
        dead_dsts: &mut [Self::Owned],
        out: &mut Vec<Self::Activity>,
    );
}

/// Reusable, lifetime-parameterized command batch collecting source and neighbor [`Buffer::Id`]s
/// for a single simulation tick.
///
/// Recycled across ticks via [`recycle`](Self::recycle) to retain heap allocations
/// without violating borrow-checker lifetimes.
pub struct Batch<'a, B: Buffer> {
    alive_srcs: Vec<B::Id<'a>>,
    alive_nbhds: Vec<NeighborData<B::Id<'a>>>,
    dead_nbhds: Vec<NeighborData<B::Id<'a>>>,
}

impl<'a, B: Buffer> Default for Batch<'a, B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, B: Buffer> Batch<'a, B> {
    /// Creates an empty batch with no heap allocations.
    pub fn new() -> Self {
        Self {
            alive_srcs: Vec::new(),
            alive_nbhds: Vec::new(),
            dead_nbhds: Vec::new(),
        }
    }

    /// Returns the number of alive chunks queued in this batch.
    pub fn alive_len(&self) -> usize {
        self.alive_srcs.len()
    }

    /// Returns the number of dead border chunks queued in this batch.
    pub fn dead_len(&self) -> usize {
        self.dead_nbhds.len()
    }

    pub fn alive_srcs(&self) -> &[B::Id<'a>] {
        &self.alive_srcs
    }

    pub fn alive_nbhds(&self) -> &[NeighborData<B::Id<'a>>] {
        &self.alive_nbhds
    }

    pub fn dead_nbhds(&self) -> &[NeighborData<B::Id<'a>>] {
        &self.dead_nbhds
    }

    /// Pre-allocates capacity for at least `additional` alive chunk entries.
    pub fn reserve_alive(&mut self, additional: usize) {
        self.alive_srcs.reserve(additional);
        self.alive_nbhds.reserve(additional);
    }

    /// Appends an active chunk and its 8 neighbors if spare capacity is available.
    pub fn push_alive_within_capacity(&mut self, src: B::Id<'a>, nbhd: NeighborData<B::Id<'a>>) {
        let _ = self.alive_srcs.push_within_capacity(src);
        let _ = self.alive_nbhds.push_within_capacity(nbhd);
    }

    /// Appends an inactive border chunk's 8 neighbors to check for cell birth.
    pub fn push_dead(&mut self, nbhd: NeighborData<B::Id<'a>>) {
        self.dead_nbhds.push(nbhd);
    }

    /// Clears all queued entries and rebinds the batch's lifetime parameter `'a` to `'b`,
    /// reusing the underlying `Vec` heap allocations across simulation steps.
    pub fn recycle<'b>(self) -> Batch<'b, B> {
        unsafe {
            Batch {
                alive_srcs: recycle_vec(self.alive_srcs),
                alive_nbhds: recycle_vec(self.alive_nbhds),
                dead_nbhds: recycle_vec(self.dead_nbhds),
            }
        }
    }
}

/// Clears a `Vec<T>` and reinterprets its raw allocation as `Vec<U>` when `T` and `U`
/// have identical size and alignment.
unsafe fn recycle_vec<T, U>(mut a: Vec<T>) -> Vec<U> {
    const {
        assert!(size_of::<T>() == size_of::<U>());
        assert!(align_of::<T>() == align_of::<U>());
    };

    a.clear();
    let (ptr, len, cap) = a.into_raw_parts();
    unsafe { Vec::from_raw_parts(ptr.cast(), len, cap) }
}
