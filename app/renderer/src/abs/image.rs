use ash::vk;
use image::GenericImageView;
use space_alloc::BuddyAllocation;

use crate::{get_device, get_memory};

use super::memory::{GpuAllocation, GpuMemory};

/// Used for swapchains, whose images shouldn't be deleted by program code.
/// Exactly equal to [GpuImage], except that the image is not destroyed on [Drop].
#[derive(Debug)]
pub struct DriverManagedGpuImage {
    pub raw: vk::Image,
    pub allocation: Option<GpuAllocation<BuddyAllocation>>,
    pub info: vk::ImageCreateInfo,
}

#[derive(Debug)]
pub struct GpuImage {
    pub raw: vk::Image,
    pub allocation: Option<GpuAllocation<BuddyAllocation>>,
    pub info: vk::ImageCreateInfo,
}

impl Drop for GpuImage {
    fn drop(&mut self) {
        unsafe {
            log::debug!(
                "dropped image handle {:?}\n{}",
                self.raw,
                std::backtrace::Backtrace::capture()
            );
            if let Some(allocation) = &self.allocation {
                get_memory().free(allocation);
            }
            get_device().destroy_image(self.raw, None);
        }
    }
}

#[derive(Debug)]
pub struct GpuTexture<Image = GpuImage> {
    pub image: Image,
    pub view: vk::ImageView,
}

impl<Image> Drop for GpuTexture<Image> {
    fn drop(&mut self) {
        unsafe {
            get_device().destroy_image_view(self.view, None);
        }
    }
}

#[derive(Clone, Copy)]
pub struct GpuTextureHandle(pub(crate) usize);
