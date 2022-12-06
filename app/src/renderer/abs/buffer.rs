use std::collections::HashMap;

use super::memory::OutOfMemory;

use super::{
    buddy::BuddyAllocator,
    cx::Cx,
    memory::{level_count, log2_ceil, Allocator, GpuAllocation, GpuMemory, MemoryUsage},
};
use ash::vk;

#[derive(Debug)]
pub struct Buffer {
    pub raw: vk::Buffer,
    pub info: vk::BufferCreateInfo,
    pub allocation: GpuAllocation,
}

impl Buffer {
    pub unsafe fn destroy(self, cx: &mut Cx, memory: &mut GpuMemory) -> anyhow::Result<()> {
        memory.free_buffer(self.allocation)?;
        cx.device.destroy_buffer(self.raw, None);
        Ok(())
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
        memory: &mut GpuMemory,
        size: u64,
    ) -> anyhow::Result<BufferSlice> {
        assert!(size <= self.info.size);

        for (buffer, allocator) in &mut self.buffers {
            match allocator.allocate(buffer.allocation.size, size, self.alignment) {
                Ok((offset, size)) => {
                    return Ok((BufferSlice {
                        buffer: buffer.raw,
                        offset,
                        size,
                    }))
                }
                Err(e) if !e.is::<OutOfMemory>() => {
                    return Err(e);
                }
                _ => {}
            }
        }

        self.buffers.push((
            memory.allocate_buffer(self.info, self.usage, self.mapped)?,
            self.default_allocator.clone(),
        ));

        self.suballocate(memory, size)
    }
}

pub struct BufferSlice {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceAddress,
    pub size: vk::DeviceSize,
}
