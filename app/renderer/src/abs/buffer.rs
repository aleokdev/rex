use std::ffi::{CStr, CString};
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut};

use crate::device::{get_device, get_memory};

use super::memory::{GpuAllocation, GpuMemory, MemoryUsage};
use ash::extensions::ext::DebugUtils;
use ash::vk::{self, Handle};
use nonzero_ext::NonZeroAble;
use space_alloc::{BuddyAllocation, BuddyAllocator, OutOfMemory};

#[derive(Clone, Debug)]
pub struct Buffer<Allocation: space_alloc::Allocation> {
    pub raw: vk::Buffer,
    pub info: vk::BufferCreateInfo,
    pub allocation: GpuAllocation<Allocation>,
}

impl<Allocation: space_alloc::Allocation> Buffer<Allocation> {
    pub unsafe fn name(
        &self,
        device: vk::Device,
        // TODO put debugutils in ctxfr
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

impl<Allocation: space_alloc::Allocation> Drop for Buffer<Allocation> {
    fn drop(&mut self) {
        let device = get_device();
        let memory = get_memory();
        unsafe {
            memory.free(&self.allocation);
        }
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

    pub fn info_string(&self) -> String {
        let mut str = String::new();
        self.buffers
            .iter()
            .for_each(|(_, allocator)| str = format!("{}\n{:?}", str, allocator));

        str
    }

    pub unsafe fn suballocate(
        &mut self,
        memory: &GpuMemory,
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
                Err(OutOfMemory) => (),
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
}

#[derive(Debug, Clone)]
pub struct BufferSlice<Allocation: space_alloc::Allocation> {
    pub buffer: Buffer<Allocation>,
    pub offset: vk::DeviceAddress,
    pub size: vk::DeviceSize,
}
