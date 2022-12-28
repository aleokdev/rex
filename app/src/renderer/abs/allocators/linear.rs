use super::{Allocation, OutOfMemory};

/// A space allocator that allocates lineally without any packing nor space reusal.
///
/// Internally, it uses a cursor that can only move forward. Once the user tries to allocate a size
/// bigger than the space left dictated by the total block size minus the cursor offset, it will
/// fail with an out of memory error.
#[derive(Debug, Clone)]
pub struct LinearAllocator {
    /// The total available space for the allocator.
    block_size: u64,
    /// Where the allocator will allocate next, relative to the block start.
    cursor_offset: u64,
}

impl LinearAllocator {
    pub fn new(block_size: u64) -> Self {
        Self {
            block_size,
            cursor_offset: 0,
        }
    }

    /// Offsets the internal cursor by the size given.
    pub fn allocate(&mut self, size: u64) -> Result<Allocation, OutOfMemory> {
        if size > self.space_free() {
            // If what we want to allocate is bigger than our space left, we obviously cannot
            // provide the allocation required.
            Err(OutOfMemory)
        } else {
            self.cursor_offset += size;
            Ok(Allocation {
                offset: self.cursor_offset - size,
                size,
                data: todo!(), // HACK
            })
        }
    }

    /// Resets the internal cursor offset to 0, thus effectively "freeing the contents" of the
    /// allocator.
    pub fn reset(&mut self) {
        self.cursor_offset = 0;
    }

    /// Returns the space left in the allocator, calculated using the internal cursor offset.
    pub fn space_free(&mut self) -> u64 {
        self.block_size - self.cursor_offset
    }
}
