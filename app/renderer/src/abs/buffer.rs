use std::ffi::{CStr, CString};
use std::num::NonZeroU64;

use super::memory::{GpuAllocation, GpuMemory, MemoryUsage};
use ash::extensions::ext::DebugUtils;
use ash::vk::{self, Handle};
use nonzero_ext::NonZeroAble;
use space_alloc::{BuddyAllocation, BuddyAllocator};

#[derive(Debug, Clone)]
pub struct Buffer<Allocation: space_alloc::Allocation> {
    pub raw: vk::Buffer,
    pub info: vk::BufferCreateInfo,
    pub allocation: GpuAllocation<Allocation>,
}

impl<Allocation: space_alloc::Allocation> Buffer<Allocation> {
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
}

impl Buffer<BuddyAllocation> {
    pub unsafe fn destroy(
        self,
        device: &ash::Device,
        memory: &mut GpuMemory,
    ) -> anyhow::Result<()> {
        memory.free(self.allocation)?;
        device.destroy_buffer(self.raw, None);
        Ok(())
    }
}

pub struct BufferArena<Allocator: space_alloc::Allocator> {
    buffers: Vec<(Buffer<Allocator::Allocation>, Allocator)>,
    info: vk::BufferCreateInfo,
    usage: MemoryUsage,
    mapped: bool,
    alignment: NonZeroU64,
    min_alloc: u64,
    debug_name: CString,
}

impl BufferArena<BuddyAllocator> {
    pub fn new(
        info: vk::BufferCreateInfo,
        usage: MemoryUsage,
        alignment: NonZeroU64,
        mapped: bool,
        min_alloc: u64,
        debug_name: CString,
    ) -> Self {
        BufferArena {
            buffers: vec![],
            info,
            usage,
            mapped,
            alignment,
            min_alloc,
            debug_name,
        }
    }

    pub unsafe fn suballocate(
        &mut self,
        memory: &mut GpuMemory,
        device: vk::Device,
        utils: &DebugUtils,
        size: NonZeroU64,
    ) -> anyhow::Result<BufferSlice<BuddyAllocation>> {
        use space_alloc::{Allocation, Allocator};
        assert!(
            size.get() <= self.info.size,
            "space tried to suballocate ({}B) was bigger than the entire buffer itself ({}B)",
            size.get(),
            self.info.size
        );

        for (buffer, allocator) in &mut self.buffers {
            match allocator.allocate(size, self.alignment) {
                Ok(allocation) => {
                    return Ok(BufferSlice {
                        buffer: buffer.clone(),
                        offset: allocation.offset(),
                        size: allocation.size(),
                    })
                }
                Err(e) => {
                    return Err(e.into());
                }
            }
        }

        let buffer = memory.allocate_buffer(self.info, self.usage, self.mapped)?;
        buffer.name(device, utils, &self.debug_name)?;
        self.buffers.push((
            buffer,
            Allocator::from_properties(self.min_alloc, self.info.size.into_nonzero().unwrap()),
        ));

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
pub struct BufferSlice<Allocation: space_alloc::Allocation> {
    pub buffer: Buffer<Allocation>,
    pub offset: vk::DeviceAddress,
    pub size: vk::DeviceSize,
}
