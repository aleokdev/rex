use std::ffi::{c_void, CStr};

use ash::{
    extensions::ext::DebugUtils,
    vk::{self, Handle},
};

use crate::{get_debug_utils, get_device};

#[derive(Debug)]
pub struct MemoryBlock<Allocator: space_alloc::Allocator> {
    raw: vk::DeviceMemory,
    allocator: Allocator,
    mapped: *mut c_void,
}

impl<Allocator: space_alloc::Allocator> MemoryBlock<Allocator> {
    /// Creates a new MemoryBlock wrapper from underlying components.
    ///
    /// ## Safety
    /// `raw` must be a valid DeviceMemory handle, and no other handle to it must exist except the
    /// one given to this function, as [`MemoryBlock`] destroys the internal buffer on drop.
    pub unsafe fn new(raw: vk::DeviceMemory, allocator: Allocator, mapped: *mut c_void) -> Self {
        Self {
            raw,
            allocator,
            mapped,
        }
    }

    pub unsafe fn name(&self, name: &CStr) -> anyhow::Result<()> {
        get_debug_utils().set_debug_utils_object_name(
            get_device().handle(),
            &vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_handle(self.raw.as_raw())
                .object_name(name)
                .object_type(vk::ObjectType::DEVICE_MEMORY),
        )?;
        Ok(())
    }

    /// Returns the underlying [vk::DeviceMemory].
    ///
    /// ## Safety
    /// The returning object must not be freed as [`MemoryBlock`] already does that on drop.
    pub fn raw(&self) -> vk::DeviceMemory {
        self.raw
    }

    pub fn allocator(&self) -> &Allocator {
        &self.allocator
    }

    pub fn mapped(&self) -> *mut c_void {
        self.mapped
    }

    pub fn allocator_mut(&mut self) -> &mut Allocator {
        &mut self.allocator
    }
}

impl<Allocator: space_alloc::Allocator> Drop for MemoryBlock<Allocator> {
    fn drop(&mut self) {
        unsafe {
            get_device().free_memory(self.raw, None);
        }
    }
}
