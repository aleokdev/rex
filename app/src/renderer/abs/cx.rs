mod debug_callback;

use super::{
    image::{Image, Texture},
    memory::{self, GpuMemory},
    util::subresource_range,
};
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    vk,
};
use cstr::cstr;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::{
    dpi::PhysicalSize,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

pub struct SwapchainTexture {
    pub color: Texture,
    pub depth: Texture,
}

pub struct SwapchainTextures(pub Vec<SwapchainTexture>);

impl SwapchainTextures {
    pub unsafe fn destroy(&mut self, device: &ash::Device, memory: &mut GpuMemory) {
        self.0.drain(..).for_each(|img| {
            device.destroy_image_view(img.color.view, None);
            img.depth.destroy(&device, memory);
        });
    }
}

/// Represents a link to the GPU, and stores common data used by both compute and graphics.
pub struct Cx {
    pub window: Window,
    /// The width of the current swapchain, which may or may not coincide with the inner width of the window.
    pub width: u32,
    /// The height of the current swapchain, which may or may not coincide with the inner height of the window.
    pub height: u32,

    pub instance: ash::Instance,
    pub debug_utils_loader: DebugUtils,
    pub debug_callback: vk::DebugUtilsMessengerEXT,
    pub surface_loader: Surface,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub swapchain_loader: Swapchain,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: SwapchainTextures,

    pub memory: super::memory::GpuMemory,

    /// The queue used for rendering along with its index.
    pub render_queue: (vk::Queue, u32),
}

pub struct SwapchainData {
    pub swapchain: vk::SwapchainKHR,
    pub images: SwapchainTextures,
}

impl Cx {
    pub unsafe fn new(event_loop: &EventLoop<()>, width: u32, height: u32) -> anyhow::Result<Self> {
        let window = WindowBuilder::new()
            .with_title("rex")
            .with_inner_size(PhysicalSize::new(width, height))
            .build(event_loop)?;

        let layers = [cstr!("VK_LAYER_KHRONOS_validation")];
        let raw_layers = layers
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<Vec<_>>();

        let mut raw_extensions =
            ash_window::enumerate_required_extensions(window.raw_display_handle())?.to_vec();
        raw_extensions.push(DebugUtils::name().as_ptr());

        let entry = ash::Entry::linked();

        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::builder()
                .application_info(
                    &vk::ApplicationInfo::builder()
                        .application_name(cstr!("Rex"))
                        .application_version(0)
                        .engine_name(cstr!("RexEngine"))
                        .engine_version(0)
                        .api_version(vk::API_VERSION_1_2),
                )
                .enabled_layer_names(&raw_layers)
                .enabled_extension_names(&raw_extensions),
            None,
        )?;

        let debug_utils_loader = DebugUtils::new(&entry, &instance);
        let debug_callback = debug_utils_loader.create_debug_utils_messenger(
            &vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback::debug_callback)),
            None,
        )?;

        let surface_loader = Surface::new(&entry, &instance);
        let surface = ash_window::create_surface(
            &entry,
            &instance,
            window.raw_display_handle(),
            window.raw_window_handle(),
            None,
        )?;

        let physical_device = instance
            .enumerate_physical_devices()?
            .into_iter()
            .max_by_key(|&physical_device| -> u64 {
                // pick GPU by the greatest local memory
                instance
                    .get_physical_device_memory_properties(physical_device)
                    .memory_heaps
                    .iter()
                    .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                    .map(|heap| heap.size)
                    .sum()
            })
            .ok_or_else(|| anyhow::anyhow!("no GPUs on this system"))?;

        let _limits = instance
            .get_physical_device_properties(physical_device)
            .limits;

        let surface_formats =
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?;
        let surface_format = surface_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .unwrap_or(surface_formats[0]);

        let mut queue = None;

        instance
            .get_physical_device_queue_family_properties(physical_device)
            .iter()
            .enumerate()
            .for_each(|(i, q)| {
                if queue.is_none() && q.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    queue = Some(i);
                }
            });

        let queue = queue.ok_or_else(|| anyhow::anyhow!("no graphics queue"))? as u32;

        let raw_device_extensions = [Swapchain::name().as_ptr()];
        let features = vk::PhysicalDeviceFeatures::default();

        let device = instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::builder()
                .queue_create_infos(
                    &[queue]
                        .into_iter()
                        .map(|queue| {
                            *vk::DeviceQueueCreateInfo::builder()
                                .queue_family_index(queue)
                                .queue_priorities(&[1.])
                        })
                        .collect::<Vec<_>>(),
                )
                .enabled_extension_names(&raw_device_extensions)
                .enabled_features(&features),
            None,
        )?;

        let queue = (device.get_device_queue(queue, 0), queue);

        let swapchain_loader = Swapchain::new(&instance, &device);

        let mut memory = super::memory::GpuMemory::new(&device, &instance, physical_device)?;

        let SwapchainData {
            swapchain,
            images: swapchain_images,
        } = Self::create_swapchain(
            &device,
            &mut memory,
            surface_format,
            &surface_loader,
            physical_device,
            surface,
            &swapchain_loader,
            width,
            height,
            vk::SwapchainKHR::null(),
        )?;

        Ok(Cx {
            window,
            width,
            height,

            swapchain_loader,
            instance,
            debug_utils_loader,
            debug_callback,
            surface_loader,
            surface,
            surface_format,
            physical_device,
            device,
            swapchain,
            swapchain_images,

            memory,

            render_queue: queue,
        })
    }

    pub unsafe fn recreate_swapchain(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        let SwapchainData { swapchain, images } = Self::create_swapchain(
            &self.device,
            &mut self.memory,
            self.surface_format,
            &self.surface_loader,
            self.physical_device,
            self.surface,
            &self.swapchain_loader,
            width,
            height,
            self.swapchain,
        )?;

        let old_swapchain = std::mem::replace(&mut self.swapchain, swapchain);
        let mut old_swapchain_images = std::mem::replace(&mut self.swapchain_images, images);

        self.device.device_wait_idle()?;

        old_swapchain_images.destroy(&self.device, &mut self.memory);
        self.swapchain_loader.destroy_swapchain(old_swapchain, None);

        self.width = width;
        self.height = height;

        Ok(())
    }

    unsafe fn create_swapchain(
        device: &ash::Device,
        memory: &mut memory::GpuMemory,
        surface_format: vk::SurfaceFormatKHR,
        surface_loader: &ash::extensions::khr::Surface,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        swapchain_loader: &ash::extensions::khr::Swapchain,
        width: u32,
        height: u32,
        old_swapchain: vk::SwapchainKHR,
    ) -> anyhow::Result<SwapchainData> {
        let surface_caps =
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?;

        let mut min_image_count = surface_caps.min_image_count + 1;
        if surface_caps.max_image_count > 0 {
            min_image_count = min_image_count.min(surface_caps.max_image_count);
        }

        let pre_transform = if surface_caps
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_caps.current_transform
        };

        let present_mode = surface_loader
            .get_physical_device_surface_present_modes(physical_device, surface)?
            .into_iter()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(vk::Extent2D { width, height })
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .old_swapchain(old_swapchain)
            .clipped(true);

        let swapchain = swapchain_loader.create_swapchain(&info, None)?;
        let swapchain_images = swapchain_loader
            .get_swapchain_images(swapchain)?
            .into_iter()
            .map(|image| -> anyhow::Result<_> {
                let extent = vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                };
                let color = Image {
                    raw: image,
                    allocation: None,
                    // synthesise some assumed information about the swapchain images
                    info: vk::ImageCreateInfo::builder()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(surface_format.format)
                        .extent(extent)
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .build(),
                };

                let color_view = device.create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .image(color.raw)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .subresource_range(subresource_range(
                            vk::ImageAspectFlags::COLOR,
                            0..1,
                            0..1,
                        )),
                    None,
                )?;

                let depth = memory.allocate_image(
                    &vk::ImageCreateInfo::builder()
                        .image_type(vk::ImageType::TYPE_2D)
                        .extent(extent)
                        .format(vk::Format::D32_SFLOAT)
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
                )?;

                let depth_view = device.create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .image(depth.raw)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::D32_SFLOAT)
                        .subresource_range(subresource_range(
                            vk::ImageAspectFlags::DEPTH,
                            0..1,
                            0..1,
                        )),
                    None,
                )?;

                Ok(SwapchainTexture {
                    color: Texture {
                        image: color,
                        view: color_view,
                    },
                    depth: Texture {
                        image: depth,
                        view: depth_view,
                    },
                })
            })
            .collect::<anyhow::Result<_>>()?;

        Ok(SwapchainData {
            swapchain,
            images: SwapchainTextures(swapchain_images),
        })
    }
}

impl Drop for Cx {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle();
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.swapchain_images
                .destroy(&self.device, &mut self.memory);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
