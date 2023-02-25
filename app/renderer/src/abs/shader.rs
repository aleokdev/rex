use std::ffi::CStr;

use ash::{
    extensions::ext::DebugUtils,
    vk::{self, Handle},
};

use crate::{get_debug_utils, get_device};

#[derive(Clone, Copy)]
pub struct ShaderModule(pub vk::ShaderModule);

impl ShaderModule {
    pub unsafe fn from_spirv_bytes(bytes: &[u8], device: &ash::Device) -> anyhow::Result<Self> {
        let vertex_code = bytes
            .chunks(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        Ok(Self(device.create_shader_module(
            &vk::ShaderModuleCreateInfo::builder().code(&vertex_code),
            None,
        )?))
    }
    pub unsafe fn name(self, name: &CStr) -> anyhow::Result<Self> {
        get_debug_utils().set_debug_utils_object_name(
            get_device().handle(),
            &vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_handle(self.0.as_raw())
                .object_name(name)
                .object_type(vk::ObjectType::SHADER_MODULE),
        )?;
        Ok(self)
    }
}
