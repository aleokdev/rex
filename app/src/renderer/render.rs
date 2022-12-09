use std::ffi::CStr;

use super::abs::{self, memory::GpuMemory};
use ash::vk;

pub struct Frame {
    pub present_semaphore: vk::Semaphore,
    pub render_semaphore: vk::Semaphore,
    pub render_fence: vk::Fence,
    pub cmd: vk::CommandBuffer,
    pub allocator: GpuMemory,
    pub deletion: Vec<Box<dyn FnOnce(&mut abs::Cx)>>,
}

impl Frame {
    pub unsafe fn new(cx: &mut abs::Cx) -> anyhow::Result<Self> {
        let render_fence = cx.device.create_fence(
            &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
            None,
        )?;

        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let present_semaphore = cx.device.create_semaphore(&semaphore_info, None)?;
        let render_semaphore = cx.device.create_semaphore(&semaphore_info, None)?;

        Ok(Frame {
            present_semaphore,
            render_semaphore,
            render_fence,
            cmd: vk::CommandBuffer::null(),
            allocator: GpuMemory::new(&cx.device, &cx.instance, cx.physical_device)?,
            deletion: vec![],
        })
    }
}

pub struct Renderer {
    pub memory: abs::memory::GpuMemory,
    pub arenas: Arenas,
    pub cmd_pool: vk::CommandPool,
    pub free_cmds: Vec<vk::CommandBuffer>,
    pub used_cmds: Vec<vk::CommandBuffer>,
    pub frame: u64,

    pub vertex_shader: vk::ShaderModule,
    pub fragment_shader: vk::ShaderModule,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,

    pub pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,

    pub frames: [Frame; Self::FRAME_OVERLAP],
}

impl Renderer {
    const FRAME_OVERLAP: usize = 2;

    pub unsafe fn new(cx: &mut abs::Cx) -> anyhow::Result<Self> {
        let memory = abs::memory::GpuMemory::new(&cx.device, &cx.instance, cx.physical_device)?;
        let arenas = Arenas::new(
            &cx.instance
                .get_physical_device_properties(cx.physical_device)
                .limits,
        );

        let cmd_pool = cx
            .device
            .create_command_pool(&vk::CommandPoolCreateInfo::builder(), None)?;

        let color_attachment = vk::AttachmentDescription::builder()
            .format(cx.surface_format.format)
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
        let pass = cx.device.create_render_pass(&renderpass_info, None)?;

        let framebuffers: Vec<vk::Framebuffer> = Self::create_swapchain_framebuffers(
            &cx.device,
            &cx.swapchain_images,
            pass,
            cx.width,
            cx.height,
        )?;

        let vertex_code = include_bytes!("../../res/tri.vert.spv")
            .chunks(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let vertex_shader = cx.device.create_shader_module(
            &vk::ShaderModuleCreateInfo::builder().code(&vertex_code),
            None,
        )?;

        let fragment_code = include_bytes!("../../res/tri.frag.spv")
            .chunks(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let fragment_shader = cx.device.create_shader_module(
            &vk::ShaderModuleCreateInfo::builder().code(&fragment_code),
            None,
        )?;

        let pipeline_layout = cx
            .device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)?;

        const ENTRY_POINT: &CStr = cstr::cstr!("main");
        let stages = &[
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader)
                .name(ENTRY_POINT)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader)
                .name(ENTRY_POINT)
                .build(),
        ];
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .cull_mode(vk::CullModeFlags::NONE)
            .depth_bias_clamp(0.)
            .depth_bias_constant_factor(0.)
            .depth_bias_slope_factor(0.)
            .depth_bias_enable(false)
            .line_width(1.)
            .front_face(vk::FrontFace::CLOCKWISE)
            .polygon_mode(vk::PolygonMode::FILL);

        let msaa = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .min_sample_shading(1.)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachments = &[*vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)];

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(color_blend_attachments);

        let viewports = &[*vk::Viewport::builder()
            .width(cx.width as f32)
            .height(cx.height as f32)
            .min_depth(0.)
            .max_depth(1.)];

        let scissors = &[*vk::Rect2D::builder().extent(vk::Extent2D {
            width: cx.width,
            height: cx.height,
        })];

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewports)
            .scissors(scissors);

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .input_assembly_state(&input_assembly_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&msaa)
            .render_pass(pass)
            .subpass(0)
            .layout(pipeline_layout)
            .color_blend_state(&color_blend_state)
            .viewport_state(&viewport_state)
            .vertex_input_state(&vertex_input_state);

        let pipeline = cx
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[*pipeline_info], None)
            .map_err(|(_, e)| e)?[0];

        Ok(Renderer {
            memory,
            arenas,
            cmd_pool,
            free_cmds: vec![],
            used_cmds: vec![],
            frame: 0,

            vertex_shader,
            fragment_shader,
            pipeline,
            pipeline_layout,

            pass,
            framebuffers,

            frames: [Frame::new(cx)?, Frame::new(cx)?],
        })
    }

    pub unsafe fn resize(&mut self, cx: &abs::Cx, width: u32, height: u32) -> anyhow::Result<()> {
        self.framebuffers
            .iter()
            .for_each(|&fb| cx.device.destroy_framebuffer(fb, None));

        self.framebuffers = Self::create_swapchain_framebuffers(
            &cx.device,
            &cx.swapchain_images,
            self.pass,
            width,
            height,
        )?;

        Ok(())
    }

    unsafe fn create_swapchain_framebuffers(
        device: &ash::Device,
        swapchain_images: &[abs::image::Texture],
        renderpass: vk::RenderPass,
        width: u32,
        height: u32,
    ) -> ash::prelude::VkResult<Vec<vk::Framebuffer>> {
        swapchain_images
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

                device.create_framebuffer(&framebuffer_info, None)
            })
            .collect()
    }

    pub unsafe fn draw(&mut self, cx: &mut abs::Cx) -> anyhow::Result<()> {
        // - Wait for the render queue fence:
        //      We don't want to submit to the queue while it is busy!
        // - Reset it:
        //      Now that we've finished waiting, the queue is free again, we can reset the
        //      fence.
        // - Obtain the next image we should be drawing onto:
        //      Since we're using a swapchain, the driver should tell us which image of the
        //      swapchain we should be drawing onto; We can't draw directly onto the window.
        // - Reset the command buffer:
        //      We used it last frame (unless this is the first frame, in which case it is
        //      already reset), now that the queue submit is finished we can safely reset it.
        // - Record commands to the buffer:
        //      We now are ready to tell the GPU what to do.
        //      - Begin a render pass:
        //          We clear the whole frame with black.
        //      - Bind our pipeline:
        //          We tell the GPU to configure itself for what's coming...
        //      - Draw:
        //          We draw our mesh having set the pipeline first.
        //      - End the render pass
        // - Submit the command buffer to the queue:
        //      We set it to take place after the image acquisition has been finalized.
        //      This operation will take a while, so we set it to open our render queue fence
        //      once it is complete.
        // - Present the frame drawn:
        //      We adjust this operation to take place after the submission has finished.
        //
        // And thus, our timeline will look something like this:
        // [        CPU        ][        GPU        ]
        // [ Setup GPU work    -> Wait for work     ]
        // [ Wait for fence    ][ Acquire image #1  ] Note: The image acquisition may stall!
        // [ Wait for fence    ][ Execute commands  ] After all the images in the swapchain
        // [ Wait for fence    <- Signal fence      ] have been written the GPU will need to
        // [ Setup GPU work    -> Present image #1  ] wait for the next refresh cycle for a
        // [ Wait for fence    ][ Acquire image #2  ] new image to be available.

        let cmd = self.acquire_cmd_buffer(cx)?;
        let frame = &mut self.frames[self.frame as usize % self.frames.len()];
        frame.cmd = cmd;
        frame.deletion.drain(..).for_each(|f| f(cx));
        frame.allocator.free_scratch()?;

        cx.device.wait_for_fences(
            &[frame.render_fence],
            true,
            std::time::Duration::from_secs(1).as_nanos() as u64,
        )?;

        cx.device.reset_fences(&[frame.render_fence])?;

        let (swapchain_img_index, _is_suboptimal) = cx.swapchain_loader.acquire_next_image(
            cx.swapchain,
            std::time::Duration::from_secs(1).as_nanos() as u64,
            frame.present_semaphore,
            ash::vk::Fence::null(),
        )?;

        cx.device.begin_command_buffer(
            frame.cmd,
            &ash::vk::CommandBufferBeginInfo::builder()
                .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        cx.device.cmd_begin_render_pass(
            frame.cmd,
            &ash::vk::RenderPassBeginInfo::builder()
                .clear_values(&[ash::vk::ClearValue {
                    color: ash::vk::ClearColorValue {
                        float32: [0., 0., 0., 1.],
                    },
                }])
                .render_area(
                    ash::vk::Rect2D::builder()
                        .extent(ash::vk::Extent2D {
                            width: cx.width,
                            height: cx.height,
                        })
                        .build(),
                )
                .render_pass(self.pass)
                .framebuffer(self.framebuffers[swapchain_img_index as usize]),
            ash::vk::SubpassContents::INLINE,
        );

        cx.device.cmd_bind_pipeline(
            frame.cmd,
            ash::vk::PipelineBindPoint::GRAPHICS,
            self.pipeline,
        );
        cx.device.cmd_draw(frame.cmd, 3, 1, 0, 0);

        cx.device.cmd_end_render_pass(frame.cmd);
        cx.device.end_command_buffer(frame.cmd)?;

        let wait_stage = &[ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let wait_semaphores = &[frame.present_semaphore];
        let signal_semaphores = &[frame.render_semaphore];
        let cmd_buffers = &[frame.cmd];
        let submit = ash::vk::SubmitInfo::builder()
            .wait_dst_stage_mask(wait_stage)
            .wait_semaphores(wait_semaphores)
            .signal_semaphores(signal_semaphores)
            .command_buffers(cmd_buffers)
            .build();

        cx.device
            .queue_submit(cx.render_queue.0, &[submit], frame.render_fence)?;

        cx.swapchain_loader.queue_present(
            cx.render_queue.0,
            &ash::vk::PresentInfoKHR::builder()
                .swapchains(&[cx.swapchain])
                .wait_semaphores(&[frame.render_semaphore])
                .image_indices(&[swapchain_img_index]),
        )?;

        Ok(())
    }

    pub unsafe fn acquire_cmd_buffer(
        &mut self,
        cx: &mut abs::Cx,
    ) -> anyhow::Result<vk::CommandBuffer> {
        self.frame += 1;

        if self.frame % 128 == 0 {
            self.free_cmds = self.used_cmds.drain(..).collect();
            cx.device
                .reset_command_pool(self.cmd_pool, Default::default())?;
        }

        if self.free_cmds.is_empty() {
            self.free_cmds = cx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_buffer_count(128)
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )?;
        }

        let cmd = self.free_cmds.pop().unwrap();
        self.used_cmds.push(cmd);
        Ok(cmd)
    }

    pub unsafe fn destroy(&self, cx: &mut abs::Cx) -> anyhow::Result<()> {
        cx.device.device_wait_idle().unwrap();

        cx.device
            .reset_command_pool(self.cmd_pool, Default::default())?;

        self.frames.iter().for_each(|frame| {
            cx.device.destroy_fence(frame.render_fence, None);
            cx.device.destroy_semaphore(frame.present_semaphore, None);
            cx.device.destroy_semaphore(frame.render_semaphore, None);
        });

        cx.device
            .destroy_pipeline_layout(self.pipeline_layout, None);
        cx.device.destroy_pipeline(self.pipeline, None);
        cx.device.destroy_shader_module(self.vertex_shader, None);
        cx.device.destroy_shader_module(self.fragment_shader, None);
        cx.device.destroy_command_pool(self.cmd_pool, None);
        cx.device.destroy_render_pass(self.pass, None);
        self.framebuffers
            .iter()
            .for_each(|&fb| cx.device.destroy_framebuffer(fb, None));

        Ok(())
    }
}

pub struct Arenas {
    pub vertex: abs::buffer::BufferArena,
    pub index: abs::buffer::BufferArena,
    pub uniform: abs::buffer::BufferArena,
}

impl Arenas {
    pub fn new(limits: &vk::PhysicalDeviceLimits) -> Self {
        Arenas {
            vertex: abs::buffer::BufferArena::new_list(
                vk::BufferCreateInfo::builder()
                    .size(64 * 1024 * std::mem::size_of::<abs::mesh::GpuVertex>() as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                abs::memory::MemoryUsage::Gpu,
                1,
                false,
                256 * std::mem::size_of::<abs::mesh::GpuVertex>() as u64,
            ),
            index: abs::buffer::BufferArena::new_list(
                vk::BufferCreateInfo::builder()
                    .size(64 * 1024 * std::mem::size_of::<u32>() as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                abs::memory::MemoryUsage::Gpu,
                1,
                false,
                256 * std::mem::size_of::<u32>() as u64,
            ),
            uniform: abs::buffer::BufferArena::new_list(
                vk::BufferCreateInfo::builder()
                    .size(64 * 128 * 1024)
                    .usage(
                        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::UNIFORM_BUFFER,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                abs::memory::MemoryUsage::Gpu,
                limits.min_uniform_buffer_offset_alignment,
                false,
                128,
            ),
        }
    }
}
