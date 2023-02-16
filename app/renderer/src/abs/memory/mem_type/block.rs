use std::ffi::{c_void, CStr};

use ash::{
    extensions::ext::DebugUtils,
    vk::{self, Handle},
};

#[derive(Debug)]
pub struct MemoryBlock<Allocator: space_alloc::Allocator> {
    pub raw: vk::DeviceMemory,
    pub allocator: Allocator,
    pub mapped: *mut c_void,
}

impl<Allocator: space_alloc::Allocator> MemoryBlock<Allocator> {
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
                .object_type(vk::ObjectType::DEVICE_MEMORY),
        )?;
        Ok(())
    }
}
