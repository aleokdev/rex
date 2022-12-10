use std::borrow::Borrow;

use super::memory::OutOfMemory;
use super::{
    buddy::BuddyAllocator,
    cx::Cx,
    memory::{level_count, log2_ceil, Allocator, GpuAllocation, GpuMemory, MemoryUsage},
};
use ash::vk;

/// Represents a valid GPU memory buffer.
#[derive(Clone)]
pub struct Buffer {
    cx: std::sync::Arc<Cx>,

    raw: vk::Buffer,
    info: vk::BufferCreateInfo,
    allocation: GpuAllocation,
}

impl Buffer {
    /// ## Safety
    /// The caller guarantees that the data passed is correct and valid, and no other object
    /// possesses it.
    pub unsafe fn from_raw_data(
        cx: std::sync::Arc<Cx>,
        raw: vk::Buffer,
        info: vk::BufferCreateInfo,
        allocation: GpuAllocation,
    ) -> Self {
        Self {
            cx,
            raw,
            info,
            allocation,
        }
    }

    /// ## Safety
    /// The caller must not destroy the buffer given.
    pub unsafe fn raw(&self) -> vk::Buffer {
        self.raw
    }

    pub fn info(&self) -> vk::BufferCreateInfo {
        self.info
    }

    pub fn allocation(&self) -> &GpuAllocation {
        &self.allocation
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            // FIXME: This causes an assertion check in the buddy allocator when destroying the renderer
            // (assert_eq!(self.block(index - 1).order_free, 0); in deallocate)
            // So we skip freeing and just destroy the buffer altogether

            // memory.free_buffer(self.allocation)?;
            self.cx.device.destroy_buffer(self.raw, None);
        }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("raw", &self.raw)
            .field("info", &self.info)
            .field("allocation", &self.allocation)
            .finish()
    }
}

pub struct BufferArena {
    buffers: Vec<(Buffer, Allocator)>,
    info: vk::BufferCreateInfo,
    usage: MemoryUsage,
    mapped: bool,
    default_allocator: Allocator,
    alignment: u64,
}

impl BufferArena {
    pub fn new_linear(
        info: vk::BufferCreateInfo,
        usage: MemoryUsage,
        mapped: bool,
        alignment: u64,
    ) -> Self {
        BufferArena {
            buffers: vec![],
            info,
            usage,
            mapped,
            default_allocator: Allocator::Linear { cursor: 0 },
            alignment,
        }
    }

    pub fn new_list(
        info: vk::BufferCreateInfo,
        usage: MemoryUsage,
        alignment: u64,
        mapped: bool,
        min_alloc: u64,
    ) -> Self {
        BufferArena {
            buffers: vec![],
            info,
            usage,
            mapped,
            default_allocator: Allocator::Buddy(BuddyAllocator::new(
                level_count(info.size, min_alloc),
                log2_ceil(min_alloc) as u8,
            )),
            alignment,
        }
    }

    pub unsafe fn suballocate(
        &mut self,
        cx: std::sync::Arc<Cx>,
        memory: &mut GpuMemory,
        size: u64,
    ) -> anyhow::Result<BufferSlice> {
        assert!(size <= self.info.size);

        for (buffer, allocator) in &mut self.buffers {
            match allocator.allocate(buffer.allocation.size, size, self.alignment) {
                Ok((offset, size)) => {
                    return Ok(BufferSlice {
                        buffer,
                        offset,
                        size,
                    })
                }
                Err(e) if !e.is::<OutOfMemory>() => {
                    return Err(e);
                }
                _ => {}
            }
        }

        self.buffers.push((
            memory.allocate_buffer(cx, self.info, self.usage, self.mapped)?,
            self.default_allocator.clone(),
        ));

        self.suballocate(cx, memory, size)
    }
}

#[derive(Debug, Clone)]
pub struct BufferSlice<'b> {
    pub buffer: &'b Buffer,
    pub offset: vk::DeviceAddress,
    pub size: vk::DeviceSize,
}

#[derive(Debug, Clone)]
pub struct OwnedBufferSlice {
    pub buffer: Buffer,
    pub offset: vk::DeviceAddress,
    pub size: vk::DeviceSize,
}

impl OwnedBufferSlice {
    pub fn borrow<'b>(&'b self) -> BufferSlice<'b> {
        BufferSlice {
            buffer: &self.buffer,
            offset: self.offset,
            size: self.size,
        }
    }
}
