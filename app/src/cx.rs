use crate::{
    image::{Image, Texture},
    memory::GpuMemory,
    util::subresource_range,
};
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    vk,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::{
    borrow::Cow,
    ffi::{CStr, CString},
};
use winit::{
    dpi::PhysicalSize,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

pub struct Cx {
    pub window: Window,
    pub width: u32,
    pub height: u32,

    pub instance: ash::Instance,
    pub debug_utils_loader: DebugUtils,
    pub debug_callback: vk::DebugUtilsMessengerEXT,
    pub surface_loader: Surface,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: (vk::Queue, u32),
    pub swapchain_loader: Swapchain,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<Texture>,

    pub memory: GpuMemory,
    pub frame: u64,

    pub renderpass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub present_semaphore: vk::Semaphore,
    pub render_semaphore: vk::Semaphore,
    pub render_fence: vk::Fence,
}

pub struct SwapchainData {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<Texture>,
}

impl Cx {
    pub fn new(event_loop: &EventLoop<()>, width: u32, height: u32) -> anyhow::Result<Self> {
        let window = WindowBuilder::new()
            .with_title("rex")
            .with_inner_size(PhysicalSize::new(width, height))
            .build(event_loop)?;

        unsafe {
            let layers = [CString::new("VK_LAYER_KHRONOS_validation")?];
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
                            .application_name(&CString::new("Rex")?)
                            .application_version(0)
                            .engine_name(&CString::new("RexEngine")?)
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
                    .pfn_user_callback(Some(debug_callback)),
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

            let surface_formats =
                surface_loader.get_physical_device_surface_formats(physical_device, surface)?;
            let surface_format = surface_formats
                .iter()
                .find(|format| {
                    format.format == vk::Format::B8G8R8A8_SRGB
                        && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .cloned()
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

            let memory = GpuMemory::new(&device, &instance, physical_device)?;

            let swapchain_loader = Swapchain::new(&instance, &device);

            let SwapchainData {
                swapchain,
                images: swapchain_images,
            } = Self::create_swapchain(
                &instance,
                &device,
                surface_format,
                &surface_loader,
                physical_device,
                surface,
                &swapchain_loader,
                width,
                height,
            )?;

            let color_attachment = vk::AttachmentDescription::builder()
                .format(surface_format.format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .build();
            let color_attachment_ref = vk::AttachmentReference::builder()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .build();
            let subpass = vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&[color_attachment_ref])
                .build();
            let color_attachments = &[color_attachment];
            let subpasses = &[subpass];
            let renderpass_info = vk::RenderPassCreateInfo::builder()
                .attachments(color_attachments)
                .subpasses(subpasses);
            let renderpass = unsafe { device.create_render_pass(&renderpass_info, None)? };

            let framebuffers: Vec<vk::Framebuffer> = swapchain_images
                .iter()
                .map(|tex| tex.view)
                .map(|img_view| {
                    let attachments = &[img_view];
                    let framebuffer_info = vk::FramebufferCreateInfo::builder()
                        .render_pass(renderpass)
                        .attachments(attachments)
                        .width(width)
                        .height(height)
                        .layers(1);
                    let framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None) };
                    framebuffer
                })
                .collect::<ash::prelude::VkResult<Vec<vk::Framebuffer>>>()?;

            let render_fence = device.create_fence(
                &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?;
            let semaphore_info = vk::SemaphoreCreateInfo::builder();
            let present_semaphore = device.create_semaphore(&semaphore_info, None)?;
            let render_semaphore = device.create_semaphore(&semaphore_info, None)?;

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
                queue,
                swapchain,
                swapchain_images,

                renderpass,
                framebuffers,

                memory,
                frame: 0,

                render_fence,
                present_semaphore,
                render_semaphore,
            })
        }
    }

    pub unsafe fn recreate_swapchain(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        let SwapchainData { swapchain, images } = Self::create_swapchain(
            &self.instance,
            &self.device,
            self.surface_format,
            &self.surface_loader,
            self.physical_device,
            self.surface,
            &self.swapchain_loader,
            width,
            height,
        )?;

        self.swapchain = swapchain;
        self.swapchain_images = images;

        // TODO also recreate framebuffers

        Ok(())
    }

    unsafe fn create_swapchain(
        instance: &ash::Instance,
        device: &ash::Device,
        surface_format: vk::SurfaceFormatKHR,
        surface_loader: &ash::extensions::khr::Surface,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        swapchain_loader: &ash::extensions::khr::Swapchain,
        width: u32,
        height: u32,
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
            .clipped(true);

        let swapchain = swapchain_loader.create_swapchain(&info, None)?;
        let swapchain_images = swapchain_loader
            .get_swapchain_images(swapchain)?
            .into_iter()
            .map(|image| -> anyhow::Result<_> {
                let image = Image {
                    raw: image,
                    allocation: None,
                    // synthesise some assumed information about the swapchain images
                    info: vk::ImageCreateInfo::builder()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(surface_format.format)
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .build(),
                };

                let view = device.create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .image(image.raw)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .subresource_range(subresource_range(
                            vk::ImageAspectFlags::COLOR,
                            0..1,
                            0..1,
                        )),
                    None,
                )?;

                Ok(Texture { image, view })
            })
            .collect::<anyhow::Result<_>>()?;

        Ok(SwapchainData {
            swapchain,
            images: swapchain_images,
        })
    }
}

impl Drop for Cx {
    fn drop(&mut self) {
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);

            self.framebuffers
                .iter()
                .for_each(|&fb| self.device.destroy_framebuffer(fb, None));
            self.device.destroy_render_pass(self.renderpass, None);

            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    cb_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let cb_data = *cb_data;

    let name = if cb_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(cb_data.p_message_id_name).to_string_lossy()
    };

    let message = if cb_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(cb_data.p_message).to_string_lossy()
    };

    let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        log::Level::Info
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::Level::Warn
    } else {
        log::Level::Error
    };

    log::log!(level, "{:?} [{}]: {}", ty, name, message);

    vk::FALSE
}
