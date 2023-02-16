use ash::vk;
use image::GenericImageView;
use space_alloc::BuddyAllocation;

use super::memory::{GpuAllocation, GpuMemory};

#[derive(Debug)]
pub struct GpuImage {
    pub raw: vk::Image,
    pub allocation: Option<GpuAllocation<BuddyAllocation>>,
    pub info: vk::ImageCreateInfo,
}

impl GpuImage {
    pub unsafe fn destroy(
        self,
        device: &ash::Device,
        memory: &mut GpuMemory,
    ) -> anyhow::Result<()> {
        if let Some(allocation) = self.allocation {
            memory.free(allocation)?;
        }
        device.destroy_image(self.raw, None);
        Ok(())
    }
}

#[derive(Debug)]
pub struct GpuTexture {
    pub image: GpuImage,
    pub view: vk::ImageView,
}

impl GpuTexture {
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

#[derive(Clone, Copy)]
pub struct GpuTextureHandle(pub(crate) usize);
