use std::ops::Deref;

use crate::traits::{Buffer, Chunk};

pub struct Arena<B: Buffer> {
    buffer: B,
    free: Vec<B::Owned>,
}

impl<B: Buffer> Arena<B> {
    pub fn new(config: B::Config) -> Self {
        Self {
            buffer: B::new(config),
            free: Vec::new(),
        }
    }

    pub fn alloc(&mut self) -> B::Owned {
        self.free.pop().unwrap_or_else(|| {
            self.buffer.grow(1, &mut self.free);
            unsafe { self.free.pop().unwrap_unchecked() }
        })
    }

    pub fn alloc_zeroed(&mut self) -> B::Owned {
        self.own(B::Chunk::empty())
    }

    pub fn dealloc(&mut self, idx: B::Owned) {
        self.free.push(idx);
    }

    pub fn own(&mut self, chunk: B::Chunk) -> B::Owned {
        let mut idx = self.alloc();
        unsafe { self.buffer.store(&mut idx, &chunk) };
        idx
    }

    pub fn empty_clone(&self) -> Self {
        Self {
            buffer: self.buffer.empty_clone(),
            free: Vec::new(),
        }
    }
}

impl<B: Buffer> Deref for Arena<B> {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
