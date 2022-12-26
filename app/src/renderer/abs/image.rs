use ash::vk;

use super::{
    cx::Cx,
    memory::{GpuAllocation, GpuMemory},
};

#[derive(Debug)]
pub struct Image {
    pub raw: vk::Image,
    pub allocation: Option<GpuAllocation>,
    pub info: vk::ImageCreateInfo,
}

impl Image {
    pub unsafe fn destroy(
        self,
        device: &ash::Device,
        memory: &mut GpuMemory,
    ) -> anyhow::Result<()> {
        if let Some(allocation) = self.allocation {
            memory.free_image(allocation)?;
        }
        device.destroy_image(self.raw, None);
        Ok(())
    }
}

#[derive(Debug)]
pub struct Texture {
    pub image: Image,
    pub view: vk::ImageView,
}

impl Texture {
    pub unsafe fn destroy(
        self,
        device: &ash::Device,
        memory: &mut GpuMemory,
    ) -> anyhow::Result<()> {
        device.destroy_image_view(self.view, None);
        self.image.destroy(device, memory)?;
        Ok(())
    }
}
