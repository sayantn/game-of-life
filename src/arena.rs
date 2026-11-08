use std::cell::UnsafeCell;

use crate::chunk::Chunk;

pub(crate) struct ChunkRef(usize);

pub(crate) struct Arena {
    chunks: Vec<UnsafeCell<Chunk>>,
    free: Vec<usize>,
}

impl Arena {
    pub(crate) fn new() -> Self {
        Self {
            chunks: Vec::new(),
            free: Vec::new(),
        }
    }

    pub(crate) fn alloc(&mut self) -> ChunkRef {
        ChunkRef(self.free.pop().unwrap_or_else(|| {
            self.chunks.push(UnsafeCell::new(Chunk::empty()));
            self.chunks.len() - 1
        }))
    }

    pub(crate) fn dealloc(&mut self, idx: ChunkRef) {
        self.free.push(idx.0);
    }

    pub(crate) fn get<'a>(&'a self, idx: &'a ChunkRef) -> &'a Chunk {
        unsafe { self.chunks[idx.0].as_ref_unchecked() }
    }

    pub(crate) fn get_mut<'a>(&'a self, idx: &'a mut ChunkRef) -> &'a mut Chunk {
        unsafe { self.chunks[idx.0].as_mut_unchecked() }
    }

    pub(crate) fn own(&mut self, chunk: Chunk) -> ChunkRef {
        if let Some(i) = self.free.pop() {
            *self.chunks[i].get_mut() = chunk;
            ChunkRef(i)
        } else {
            self.chunks.push(UnsafeCell::new(chunk));
            ChunkRef(self.chunks.len() - 1)
        }
    }
}
