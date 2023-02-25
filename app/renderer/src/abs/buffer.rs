use std::ffi::{CStr, CString};
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut};

use crate::device::{get_device, get_memory};
use crate::get_debug_utils;

use super::memory::{GpuAllocation, GpuMemory, MemoryUsage};
use anyhow::anyhow;
use ash::extensions::ext::DebugUtils;
use ash::vk::{self, Handle};
use nonzero_ext::NonZeroAble;
use space_alloc::{BuddyAllocation, BuddyAllocator, OutOfMemory};

#[derive(Debug)]
pub struct Buffer<Allocation: space_alloc::Allocation> {
    raw: vk::Buffer,
    info: vk::BufferCreateInfo,
    allocation: GpuAllocation<Allocation>,
}

impl<Allocation: space_alloc::Allocation> Buffer<Allocation> {
    /// Creates a new Buffer wrapper from underlying components.
    ///
    /// ## Safety
    /// `raw` must be a valid Buffer handle, and no other handle to the buffer must exist except the
    /// one given to this function, as [`Buffer`] destroys the internal buffer on drop.
    pub unsafe fn new(
        raw: vk::Buffer,
        info: vk::BufferCreateInfo,
        allocation: GpuAllocation<Allocation>,
    ) -> Self {
        Self {
            raw,
            info,
            allocation,
        }
    }

    /// Returns the underlying buffer object.
    ///
    /// ## Note
    /// The returned buffer must not be destroyed, as [`Buffer`] destroys it as well on drop.
    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }

    pub fn info(&self) -> vk::BufferCreateInfo {
        self.info
    }

    pub fn allocation(&self) -> &GpuAllocation<Allocation> {
        &self.allocation
    }

    /// Create a view (or 'slice') into this buffer with the offset and size given.
    ///
    /// ## Safety
    /// This is an unsafe operation because [`BufferSlice`] has no lifetime information, and can
    /// refer to the buffer after free.
    ///
    /// This is required, at least for now, due to how [`BufferArena`] works (`suballocate` would
    /// require to return a `'self` buffer slice, but then that'd require making the method accept
    /// `&self` instead to be able to suballocate more than one buffer, and it'd also add lots of
    /// self referential problems for the renderer)
    pub unsafe fn slice(
        &self,
        offset: vk::DeviceAddress,
        size: vk::DeviceSize,
    ) -> anyhow::Result<BufferSlice> {
        if (offset + size) > self.info.size {
            return Err(anyhow!("tried to slice buffer with invalid offset/size: offset {} & size {} while buffer is only {}B long", offset, size, self.info.size));
        }

        Ok(BufferSlice {
            raw: self.raw,
            offset,
            size,
        })
    }
}

impl<Allocation: space_alloc::Allocation> Buffer<Allocation> {
    pub unsafe fn name(&self, name: &CStr) -> anyhow::Result<()> {
        get_debug_utils().set_debug_utils_object_name(
            get_device().handle(),
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
        log::debug!(
            "destroying buffer {:?}\n{}",
            self.raw,
            std::backtrace::Backtrace::capture()
        );
        let device = get_device();
        let memory = get_memory();
        unsafe {
            device.destroy_buffer(self.raw, None);
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

    /// Suballocates one of the internal buffers of the buffer arena, and returns a slice into it.
    ///
    /// The resulting slice is valid only for the duration of the arena.
    pub unsafe fn suballocate(
        &mut self,
        memory: &GpuMemory,
        device: vk::Device,
        utils: &DebugUtils,
        size: NonZeroU64,
    ) -> anyhow::Result<BufferSlice> {
        use space_alloc::{Allocation, Allocator};
        assert!(
            size.get() <= self.info.size,
            "space tried to suballocate ({}B) was bigger than the entire buffer itself ({}B)",
            size.get(),
            self.info.size
        );

        for (buffer, allocator) in &mut self.buffers {
            match allocator.allocate(size, self.alignment) {
                Ok(allocation) => return Ok(buffer.slice(allocation.offset(), allocation.size())?),
                Err(OutOfMemory) => (),
            }
        }

        let buffer = memory.allocate_buffer(self.info, self.usage, self.mapped)?;
        buffer.name(&self.debug_name)?;
        self.buffers.push((
            buffer,
            Allocator::from_properties(self.min_alloc, self.info.size.into_nonzero().unwrap()),
        ));

        self.suballocate(memory, device, utils, size)
    }
}

#[derive(Debug, Clone)]
pub struct BufferSlice {
    raw: vk::Buffer,
    offset: vk::DeviceAddress,
    size: vk::DeviceSize,
}

impl BufferSlice {
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns the underlying buffer object.
    ///
    /// ## Note
    /// The returned buffer must not be destroyed, as [`Buffer`] destroys it as well on drop.
    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }
}
