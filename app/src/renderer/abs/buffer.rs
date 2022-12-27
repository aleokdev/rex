use std::ffi::{CStr, CString};

use crate::renderer::abs::allocators::Allocation;

use super::allocators::BuddyAllocator;
use super::memory::{level_count, log2_ceil, Allocator, GpuAllocation, GpuMemory, MemoryUsage};
use ash::extensions::ext::DebugUtils;
use ash::vk::{self, Handle};

#[derive(Debug, Clone)]
pub struct Buffer {
    pub raw: vk::Buffer,
    pub info: vk::BufferCreateInfo,
    pub allocation: GpuAllocation,
}

impl Buffer {
    pub fn null() -> Self {
        Buffer {
            raw: vk::Buffer::null(),
            info: Default::default(),
            allocation: GpuAllocation::null(),
        }
    }

    pub unsafe fn name(
        &self,
        device: vk::Device,
        utils: &DebugUtils,
        name: &CStr,
    ) -> anyhow::Result<()> {
        utils.debug_utils_set_object_name(
            device,
            &vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_handle(self.raw.as_raw())
                .object_name(name)
                .object_type(vk::ObjectType::BUFFER),
        )?;
        Ok(())
    }

    pub unsafe fn destroy(
        self,
        device: &ash::Device,
        memory: &mut GpuMemory,
    ) -> anyhow::Result<()> {
        // FIXME: This causes an assertion check in the buddy allocator when destroying the renderer
        // (assert_eq!(self.block(index - 1).order_free, 0); in deallocate)
        // So we skip freeing and just destroy the buffer altogether

        // memory.free_buffer(self.allocation)?;
        device.destroy_buffer(self.raw, None);
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
    debug_name: CString,
}

impl BufferArena {
    pub fn new_linear(
        info: vk::BufferCreateInfo,
        usage: MemoryUsage,
        mapped: bool,
        alignment: u64,
        debug_name: CString,
    ) -> Self {
        BufferArena {
            buffers: vec![],
            info,
            usage,
            mapped,
            default_allocator: Allocator::Linear { cursor_offset: 0 },
            alignment,
            debug_name,
        }
    }

    pub fn new_list(
        info: vk::BufferCreateInfo,
        usage: MemoryUsage,
        alignment: u64,
        mapped: bool,
        min_alloc: u64,
        debug_name: CString,
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
            debug_name,
        }
    }

    pub unsafe fn suballocate(
        &mut self,
        memory: &mut GpuMemory,
        device: vk::Device,
        utils: &DebugUtils,
        size: u64,
    ) -> anyhow::Result<BufferSlice> {
        assert!(size <= self.info.size);

        for (buffer, allocator) in &mut self.buffers {
            match allocator.allocate(buffer.allocation.size, size, self.alignment) {
                Ok(Allocation { offset, size }) => {
                    return Ok(BufferSlice {
                        buffer: buffer.clone(),
                        offset,
                        size,
                    })
                }
                Err(e) => {
                    return Err(e.into());
                }
                _ => {}
            }
        }

        let buffer = memory.allocate_buffer(self.info, self.usage, self.mapped)?;
        buffer.name(device, utils, &self.debug_name)?;
        self.buffers.push((buffer, self.default_allocator.clone()));

        self.suballocate(memory, device, utils, size)
    }

    pub unsafe fn destroy(
        self,
        device: &ash::Device,
        memory: &mut GpuMemory,
    ) -> anyhow::Result<()> {
        for (buf, _) in self.buffers {
            buf.destroy(device, memory)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BufferSlice {
    pub buffer: Buffer,
    pub offset: vk::DeviceAddress,
    pub size: vk::DeviceSize,
}

impl BufferSlice {
    pub fn null() -> Self {
        BufferSlice {
            buffer: Buffer::null(),
            offset: 0,
            size: 0,
        }
    }
}
