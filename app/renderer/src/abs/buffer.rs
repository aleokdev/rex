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

struct BufferArena<Allocator: space_alloc::Allocator> {
    buffers: Vec<(Buffer<Allocator::Allocation>, Allocator)>,
    info: vk::BufferCreateInfo,
    usage: MemoryUsage,
    mapped: bool,
    alignment: NonZeroU64,
    min_alloc: u64,
    debug_name: CString,
}

// Need wrap to impl drop for generic instantiation.
pub struct BuddyBufferArena(BufferArena<BuddyAllocator>);

impl BuddyBufferArena {
    pub fn new(
        info: vk::BufferCreateInfo,
        usage: MemoryUsage,
        alignment: NonZeroU64,
        mapped: bool,
        min_alloc: u64,
        debug_name: CString,
    ) -> Self {
        Self(BufferArena {
            buffers: vec![],
            info,
            usage,
            mapped,
            alignment,
            min_alloc,
            debug_name,
        })
    }

    pub fn info_string(&self) -> String {
        let mut str = String::new();
        self.0
            .buffers
            .iter()
            .for_each(|(_, allocator)| str = format!("{}\n{:?}", str, allocator));

        str
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
            size.get() <= self.0.info.size,
            "space tried to suballocate ({}B) was bigger than the entire buffer itself ({}B)",
            size.get(),
            self.0.info.size
        );

        for (buffer, allocator) in &mut self.0.buffers {
            match allocator.allocate(size, self.0.alignment) {
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

        let buffer = memory.allocate_buffer(self.0.info, self.0.usage, self.0.mapped)?;
        buffer.name(device, utils, &self.0.debug_name)?;
        self.0.buffers.push((
            buffer,
            Allocator::from_properties(self.0.min_alloc, self.0.info.size.into_nonzero().unwrap()),
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
