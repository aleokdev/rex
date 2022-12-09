use std::ffi::CStr;

use super::{
    abs::{self},
    world::World,
};
use ash::vk;

pub struct Frame {
    pub present_semaphore: vk::Semaphore,
    pub render_semaphore: vk::Semaphore,
    pub render_fence: vk::Fence,
    pub cmd_pool: vk::CommandPool,
    pub cmd: vk::CommandBuffer,
    pub allocator: abs::memory::GpuMemory,
    pub deletion: Vec<Box<dyn FnOnce(&mut abs::Cx)>>,

    pub counter: u64,
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

        let cmd_pool = cx
            .device
            .create_command_pool(&vk::CommandPoolCreateInfo::builder(), None)?;

        Ok(Frame {
            present_semaphore,
            render_semaphore,
            render_fence,
            cmd_pool,
            cmd: vk::CommandBuffer::null(),
            allocator: abs::memory::GpuMemory::new(&cx.device, &cx.instance, cx.physical_device)?,
            deletion: vec![],

            counter: 0,
        })
    }
}

pub struct Renderer {
    pub memory: abs::memory::GpuMemory,
    pub ds_allocator: abs::descriptor::DescriptorAllocator,
    pub ds_layout_cache: abs::descriptor::DescriptorLayoutCache,
    pub arenas: Arenas,
    pub frame: u64,

    pub vertex_shader: vk::ShaderModule,
    pub fragment_shader: vk::ShaderModule,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,

    pub pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,

    pub cube: abs::mesh::GpuMesh,
    pub uniforms: abs::buffer::BufferSlice,
    pub uniform_set: vk::DescriptorSet,

    pub frames: [Option<Frame>; Self::FRAME_OVERLAP],
    pub deletion: Vec<Box<dyn FnOnce(&mut abs::Cx)>>,
}

impl Renderer {
    const FRAME_OVERLAP: usize = 2;

    pub unsafe fn new(cx: &mut abs::Cx) -> anyhow::Result<Self> {
        let mut memory = abs::memory::GpuMemory::new(&cx.device, &cx.instance, cx.physical_device)?;

        let mut ds_allocator = abs::descriptor::DescriptorAllocator::new(cx);
        let mut ds_layout_cache = abs::descriptor::DescriptorLayoutCache::new(cx);

        let mut arenas = Arenas::new(
            &cx.instance
                .get_physical_device_properties(cx.physical_device)
                .limits,
        );

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

        let uniforms = arenas
            .uniform
            .suballocate(&mut memory, std::mem::size_of::<[f32; 16]>() as u64)?;

        let (uniform_set, uniform_set_layout) = abs::descriptor::DescriptorBuilder::new()
            .bind_buffer(
                0,
                &[vk::DescriptorBufferInfo::builder()
                    .buffer(uniforms.buffer.raw)
                    .offset(uniforms.offset)
                    .range(std::mem::size_of::<[f32; 16]>() as u64)
                    .build()],
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::VERTEX,
            )
            .build(cx, &mut ds_allocator, &mut ds_layout_cache)?;

        let pipeline_layout = cx.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder().set_layouts(&[uniform_set_layout]),
            None,
        )?;

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

        let cube = abs::mesh::GpuMesh {
            vertices: abs::buffer::BufferSlice::null(),
            indices: abs::buffer::BufferSlice::null(),
        };

        Ok(Renderer {
            memory,
            ds_allocator,
            ds_layout_cache,
            arenas,
            frame: 0,

            vertex_shader,
            fragment_shader,
            pipeline,
            pipeline_layout,

            pass,
            framebuffers,

            cube,
            uniforms,
            uniform_set,

            frames: [Some(Frame::new(cx)?), Some(Frame::new(cx)?)],
            deletion: vec![],
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

    pub unsafe fn draw(&mut self, cx: &mut abs::Cx, world: &World) -> anyhow::Result<()> {
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

        let first = self.frame == 0;

        let mut frame = self.frames[self.frame as usize % self.frames.len()]
            .take()
            .unwrap();
        frame.deletion.drain(..).for_each(|f| f(cx));
        frame.allocator.free_scratch()?;

        cx.device.wait_for_fences(
            &[frame.render_fence],
            true,
            std::time::Duration::from_secs(1).as_nanos() as u64,
        )?;

        frame.allocator.free_scratch()?;

        if frame.counter % 128 == 0 {
            cx.device
                .reset_command_pool(frame.cmd_pool, vk::CommandPoolResetFlags::empty())?;
        }

        frame.cmd = cx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(frame.cmd_pool),
        )?[0];

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

        if first {
            self.setup(cx, &mut frame)?;
        }

        abs::memory::stage(
            &cx.device,
            &mut frame.allocator,
            frame.cmd,
            &(world.camera.proj() * world.camera.view()).to_cols_array(),
            &self.uniforms,
        )?;

        abs::memory::stage_sync(&cx.device, frame.cmd);

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

        cx.device.cmd_bind_descriptor_sets(
            frame.cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[self.uniform_set],
            &[],
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

        frame.counter += 1;
        self.frames[self.frame as usize % self.frames.len()] = Some(frame);
        self.frame += 1;

        Ok(())
    }

    pub unsafe fn setup(&mut self, cx: &mut abs::Cx, frame: &mut Frame) -> anyhow::Result<()> {
        self.cube.vertices = self.arenas.vertex.suballocate(
            &mut self.memory,
            std::mem::size_of::<[abs::mesh::GpuVertex; 3]>() as u64,
        )?;

        self.cube.indices = self
            .arenas
            .index
            .suballocate(&mut self.memory, std::mem::size_of::<[u32; 3]>() as u64)?;

        // Upload epic triangular cube as an example for now
        self.cube.upload(
            cx,
            &mut frame.allocator,
            frame.cmd,
            &[
                abs::mesh::GpuVertex {
                    position: [0., 0., 0.],
                    normal: [0., 0., 0.],
                },
                abs::mesh::GpuVertex {
                    position: [0., 0., 0.],
                    normal: [0., 0., 0.],
                },
                abs::mesh::GpuVertex {
                    position: [0., 0., 0.],
                    normal: [0., 0., 0.],
                },
            ],
            &[0, 1, 2],
        )?;

        Ok(())
    }

    pub unsafe fn destroy(mut self, cx: &mut abs::Cx) -> anyhow::Result<()> {
        cx.device.device_wait_idle().unwrap();

        self.deletion.drain(..).for_each(|f| f(cx));

        self.frames.iter().for_each(|frame| {
            let frame = frame.as_ref().unwrap();
            cx.device.destroy_fence(frame.render_fence, None);
            cx.device.destroy_semaphore(frame.present_semaphore, None);
            cx.device.destroy_semaphore(frame.render_semaphore, None);

            cx.device
                .reset_command_pool(frame.cmd_pool, Default::default())
                .unwrap();
            cx.device.destroy_command_pool(frame.cmd_pool, None);
        });

        self.ds_layout_cache.destroy();
        self.ds_allocator.destroy(cx);
        cx.device
            .destroy_pipeline_layout(self.pipeline_layout, None);
        cx.device.destroy_pipeline(self.pipeline, None);
        cx.device.destroy_shader_module(self.vertex_shader, None);
        cx.device.destroy_shader_module(self.fragment_shader, None);
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
    pub scratch: abs::buffer::BufferArena,
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
            scratch: abs::buffer::BufferArena::new_list(
                vk::BufferCreateInfo::builder()
                    .size(64 * 128 * 1024)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                abs::memory::MemoryUsage::CpuToGpu,
                1,
                true,
                128,
            ),
        }
    }
}
