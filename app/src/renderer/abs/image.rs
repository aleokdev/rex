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
    pub unsafe fn destroy(self, cx: &mut Cx, memory: &mut GpuMemory) -> anyhow::Result<()> {
        if let Some(allocation) = self.allocation {
            memory.free_image(allocation)?;
        }
        cx.device.destroy_image(self.raw, None);
        Ok(())
    }
}

#[derive(Debug)]
pub struct Texture {
    pub image: Image,
    pub view: vk::ImageView,
}

impl Texture {
    pub unsafe fn destroy(self, cx: &mut Cx, memory: &mut GpuMemory) -> anyhow::Result<()> {
        cx.device.destroy_image_view(self.view, None);
        self.image.destroy(cx, memory)?;
        Ok(())
    }
}
