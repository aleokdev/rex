use ash::vk;

use crate::{cx::Cx, memory::GpuAllocation};

#[derive(Debug)]
pub struct Image {
    pub raw: vk::Image,
    pub allocation: Option<GpuAllocation>,
    pub info: vk::ImageCreateInfo,
}

impl Image {
    pub unsafe fn destroy(self, cx: &mut Cx) {
        if let Some(allocation) = self.allocation {
            cx.memory.free(allocation);
        }
        cx.device.destroy_image(self.raw, None);
    }
}

#[derive(Debug)]
pub struct Texture {
    pub image: Image,
    pub view: vk::ImageView,
}

impl Texture {
    pub unsafe fn destroy(self, cx: &mut Cx) {
        cx.device.destroy_image_view(self.view, None);
        self.image.destroy(cx);
    }
}
