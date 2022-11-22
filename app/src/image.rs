use ash::vk;

use crate::{cx::Cx, memory::GpuAllocation};

#[derive(Debug)]
pub struct Image {
    pub raw: vk::Image,
    pub allocation: Option<GpuAllocation>,
    pub info: vk::ImageCreateInfo,
}

impl Image {
    pub unsafe fn destroy(self, cx: &mut Cx) -> anyhow::Result<()> {
        if let Some(allocation) = self.allocation {
            cx.memory.free_image(allocation)?;
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
    pub unsafe fn destroy(self, cx: &mut Cx) -> anyhow::Result<()> {
        cx.device.destroy_image_view(self.view, None);
        self.image.destroy(cx)?;
        Ok(())
    }
}
