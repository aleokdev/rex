use std::ffi::CStr;

use crate::{renderer::abs::mesh::GpuVertex, world::World};

use super::abs::{
    self,
    allocators::{BuddyAllocation, BuddyAllocator},
    memory::GpuMemory,
    Cx,
};
use ash::vk;
use nonzero_ext::{nonzero, NonZeroAble};

/// Represents an in-flight render frame.
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

        // (aleok): We have one fence per frame so only one command buffer is technically required;
        // TODO: We could use multiple so we need to reset less often but I'm leaving that to False
        // P.D: This change is justified since what we were doing previously was allocate one command
        // buffer per frame (and crash because getting out of device memory about 30k frames in)
        let cmd = cx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(cmd_pool),
        )?[0];

        Ok(Frame {
            present_semaphore,
            render_semaphore,
            render_fence,
            cmd_pool,
            cmd,
            allocator: abs::memory::GpuMemory::new(&cx.device, &cx.instance, cx.physical_device)?,
            deletion: vec![],

            counter: 0,
        })
    }
}

pub struct Renderer {
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

    pub cube: Option<abs::mesh::GpuMesh>,
    pub uniforms: abs::buffer::BufferSlice<BuddyAllocation>,
    pub uniform_set: vk::DescriptorSet,

    pub frames: [Option<Frame>; Self::FRAME_OVERLAP],
    pub deletion: Vec<Box<dyn FnOnce(&mut abs::Cx)>>,
}

impl Renderer {
    const FRAME_OVERLAP: usize = 2;

    pub unsafe fn new(cx: &mut abs::Cx) -> anyhow::Result<Self> {
        let mut ds_allocator = abs::descriptor::DescriptorAllocator::new(cx);
        let mut ds_layout_cache = abs::descriptor::DescriptorLayoutCache::new(cx);

        let mut arenas = Arenas::new(
            &cx,
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

        let depth_attachment = vk::AttachmentDescription::builder()
            .format(vk::Format::D32_SFLOAT) // TODO: Do not hardcode D322_SFLOAT for depth/stencil
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE) // We aren't using stencil for now
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        let depth_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&[color_attachment_ref])
            .depth_stencil_attachment(&depth_attachment_ref)
            .build();

        let attachments = &[color_attachment, depth_attachment];
        let subpasses = &[subpass];
        let renderpass_info = vk::RenderPassCreateInfo::builder()
            .attachments(attachments)
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

        let uniforms = arenas.uniform.suballocate(
            &mut cx.memory,
            cx.device.handle(),
            &cx.debug_utils_loader,
            (std::mem::size_of::<[f32; 16]>() as u64)
                .into_nonzero()
                .unwrap(),
        )?;

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

        let vertex_input_state = GpuVertex::description();
        let vertex_input_state_raw = vertex_input_state.raw();

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.)
            .max_depth_bounds(1.)
            .stencil_test_enable(false);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .input_assembly_state(&input_assembly_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&msaa)
            .render_pass(pass)
            .subpass(0)
            .layout(pipeline_layout)
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&depth_stencil_state)
            .viewport_state(&viewport_state)
            .vertex_input_state(&*vertex_input_state_raw);

        let pipeline = cx
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[*pipeline_info], None)
            .map_err(|(_, e)| e)?[0];

        Ok(Renderer {
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

            cube: None,
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
        swapchain_images: &abs::cx::SwapchainTextures,
        renderpass: vk::RenderPass,
        width: u32,
        height: u32,
    ) -> ash::prelude::VkResult<Vec<vk::Framebuffer>> {
        swapchain_images
            .0
            .iter()
            .map(|tex| {
                let attachments = &[tex.color.view, tex.depth.view];
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
        // Get the next frame to draw onto:
        //      We have a few in-flight frames so the GPU doesn't stay still (And we don't need to
        //      wait too much CPU-side for queue submissions to happen).
        let mut frame = self.frames[self.frame as usize % self.frames.len()]
            .take()
            .unwrap();

        // Process the deletion queue for this frame.
        frame.deletion.drain(..).for_each(|f| f(cx));

        // Free all scratch memory, since it's only used for uploading to the GPU.
        frame.allocator.free_scratch()?;

        // Wait for the frame render queue fence:
        //      We only have command buffer per frame so we need to wait for the previous one before
        //      submitting again, so we can reset it and use it once again.
        cx.device.wait_for_fences(
            &[frame.render_fence],
            true,
            std::time::Duration::from_secs(1).as_nanos() as u64,
        )?;

        // Reset the render fence:
        //      We have already waited for it, so we reset it preparing it for this next upload
        cx.device.reset_fences(&[frame.render_fence])?;

        // Reset the frame command buffer so we can reuse it:
        //      Technically we shouldn't do this, and we should have a couple command buffers to use
        //      before resetting them all (Resetting command buffers individually is a bit slower
        //      than resetting the entire pool)
        // https://arm-software.github.io/vulkan_best_practice_for_mobile_developers/samples/performance/command_buffer_usage/command_buffer_usage_tutorial.html#resetting-individual-command-buffers
        cx.device
            .reset_command_pool(frame.cmd_pool, vk::CommandPoolResetFlags::empty())?;

        // Obtain the next image we should be drawing onto:
        //      We're using a swapchain, so the driver should tell us which image of the
        //      swapchain we should be drawing onto; We can't draw directly onto the window.
        let (swapchain_img_index, _is_suboptimal) = cx.swapchain_loader.acquire_next_image(
            cx.swapchain,
            std::time::Duration::from_secs(1).as_nanos() as u64,
            frame.present_semaphore,
            ash::vk::Fence::null(),
        )?;

        // Start recording commands to the command buffer:
        //      We now are ready to tell the GPU what to do.
        cx.device.begin_command_buffer(
            frame.cmd,
            &ash::vk::CommandBufferBeginInfo::builder()
                .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        // Upload the cube before rendering if we haven't already.
        let cube = self
            .cube
            .get_or_insert_with(|| Self::setup_cube(cx, &mut self.arenas, &mut frame).unwrap());

        // Upload the camera uniform PV matrix for this frame.
        abs::memory::cmd_stage(
            &cx.device,
            &mut frame.allocator,
            frame.cmd,
            &(world.camera.proj() * world.camera.view()).to_cols_array(),
            &self.uniforms,
        )?
        .name(
            cx.device.handle(),
            &cx.debug_utils_loader,
            cstr::cstr!("Camera Uniform Scratch Buffer"),
        )?;

        // Insert a memory barrier to wait until the cube mesh and camera uniform have been uploaded.
        abs::memory::cmd_stage_sync(&cx.device, frame.cmd);

        // Begin a render pass:
        //      We clear the whole frame with black.
        cx.device.cmd_begin_render_pass(
            frame.cmd,
            &ash::vk::RenderPassBeginInfo::builder()
                .clear_values(&[
                    ash::vk::ClearValue {
                        color: ash::vk::ClearColorValue {
                            float32: [0., 0., 0., 1.],
                        },
                    },
                    ash::vk::ClearValue {
                        depth_stencil: ash::vk::ClearDepthStencilValue {
                            depth: 1.,
                            stencil: 0,
                        },
                    },
                ])
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

        // Bind our pipeline:
        //      We tell the GPU to configure its layout for what's coming...
        cx.device.cmd_bind_pipeline(
            frame.cmd,
            ash::vk::PipelineBindPoint::GRAPHICS,
            self.pipeline,
        );

        // Bind the camera uniform descriptor set and cube vertex/index buffers.
        cx.device.cmd_bind_descriptor_sets(
            frame.cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[self.uniform_set],
            &[],
        );

        cx.device.cmd_bind_index_buffer(
            frame.cmd,
            cube.indices.buffer.raw,
            0,
            vk::IndexType::UINT32,
        );
        cx.device
            .cmd_bind_vertex_buffers(frame.cmd, 0, &[cube.vertices.buffer.raw], &[0]);

        // Draw the mesh taking the index buffer into account.
        cx.device
            .cmd_draw_indexed(frame.cmd, cube.vertex_count, 1, 0, 0, 0);

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

        // Submit the command buffer to the queue:
        //      We set it to take place after the image acquisition has been finalized.
        //      This operation will take a while, so we set it to open our render queue fence
        //      once it is complete.
        cx.device
            .queue_submit(cx.render_queue.0, &[submit], frame.render_fence)?;

        // Present the frame drawn:
        //      We adjust this operation to take place after the submission has finished.
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

    unsafe fn setup_cube(
        cx: &mut abs::Cx,
        arenas: &mut Arenas,
        frame: &mut Frame,
    ) -> anyhow::Result<abs::mesh::GpuMesh> {
        let vertices = [
            abs::mesh::GpuVertex {
                position: [0., 0., 1.],
                normal: [0., 0., 0.],
            },
            abs::mesh::GpuVertex {
                position: [1., 0., 1.],
                normal: [0., 0., 0.],
            },
            abs::mesh::GpuVertex {
                position: [0., 1., 1.],
                normal: [0., 0., 0.],
            },
            abs::mesh::GpuVertex {
                position: [1., 1., 1.],
                normal: [0., 0., 0.],
            },
            abs::mesh::GpuVertex {
                position: [0., 0., 0.],
                normal: [0., 0., 0.],
            },
            abs::mesh::GpuVertex {
                position: [1., 0., 0.],
                normal: [0., 0., 0.],
            },
            abs::mesh::GpuVertex {
                position: [0., 1., 0.],
                normal: [0., 0., 0.],
            },
            abs::mesh::GpuVertex {
                position: [1., 1., 0.],
                normal: [0., 0., 0.],
            },
        ];

        let indices = [
            2, 6, 7, 2, 3, 7, //Top
            0, 4, 5, 0, 1, 5, //Bottom
            0, 2, 6, 0, 4, 6, //Left
            1, 3, 7, 1, 5, 7, //Right
            0, 2, 3, 0, 1, 3, //Front
            4, 6, 7, 4, 5, 7, //Back
        ];

        let vertices_gpu = arenas.vertex.suballocate(
            &mut cx.memory,
            cx.device.handle(),
            &cx.debug_utils_loader,
            (std::mem::size_of_val(&vertices) as u64)
                .into_nonzero()
                .unwrap(),
        )?;

        let indices_gpu = arenas.index.suballocate(
            &mut cx.memory,
            cx.device.handle(),
            &cx.debug_utils_loader,
            (std::mem::size_of_val(&indices) as u64)
                .into_nonzero()
                .unwrap(),
        )?;

        let mut cube = abs::mesh::GpuMesh {
            indices: indices_gpu,
            vertices: vertices_gpu,
            vertex_count: indices.len() as u32,
        };

        cube.upload(cx, &mut frame.allocator, frame.cmd, &vertices, &indices)?;

        Ok(cube)
    }

    pub unsafe fn destroy(mut self, cx: &mut abs::Cx) -> anyhow::Result<()> {
        cx.device.device_wait_idle().unwrap();

        self.deletion.drain(..).for_each(|f| f(cx));

        self.frames.iter_mut().for_each(|frame| {
            let frame = frame.take().unwrap();
            cx.device.destroy_fence(frame.render_fence, None);
            cx.device.destroy_semaphore(frame.present_semaphore, None);
            cx.device.destroy_semaphore(frame.render_semaphore, None);

            cx.device
                .reset_command_pool(frame.cmd_pool, Default::default())
                .unwrap();
            cx.device.destroy_command_pool(frame.cmd_pool, None);

            drop(frame.allocator);
        });

        self.arenas.destroy(&cx.device, &mut cx.memory)?;
        drop(self.ds_layout_cache);
        drop(self.ds_allocator);
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
    pub vertex: abs::buffer::BufferArena<BuddyAllocator>,
    pub index: abs::buffer::BufferArena<BuddyAllocator>,
    pub uniform: abs::buffer::BufferArena<BuddyAllocator>,
    pub scratch: abs::buffer::BufferArena<BuddyAllocator>,
}

impl Arenas {
    pub fn new(cx: &Cx, limits: &vk::PhysicalDeviceLimits) -> Self {
        Arenas {
            vertex: abs::buffer::BufferArena::new(
                vk::BufferCreateInfo::builder()
                    .size(64 * 1024 * std::mem::size_of::<abs::mesh::GpuVertex>() as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                abs::memory::MemoryUsage::Gpu,
                // From https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap22.html#fxvertex-input-extraction:
                // If format is a packed format, attribAddress must be a multiple of the size in
                // bytes of the whole attribute data type as described in Packed Formats. Otherwise,
                // attribAddress must be a multiple of the size in bytes of the component type
                // indicated by format (see Formats).
                // In our case it's unpacked and the component type size in bytes is 4.
                nonzero!(4u64),
                false,
                256 * std::mem::size_of::<abs::mesh::GpuVertex>() as u64,
                cstr::cstr!("Vertex buffer arena").to_owned(),
            ),
            index: abs::buffer::BufferArena::new(
                vk::BufferCreateInfo::builder()
                    .size(64 * 1024 * std::mem::size_of::<u32>() as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                abs::memory::MemoryUsage::Gpu,
                nonzero!(4u64),
                false,
                256 * std::mem::size_of::<u32>() as u64,
                cstr::cstr!("Index buffer arena").to_owned(),
            ),
            uniform: abs::buffer::BufferArena::new(
                vk::BufferCreateInfo::builder()
                    .size(64 * 128 * 1024)
                    .usage(
                        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::UNIFORM_BUFFER,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                abs::memory::MemoryUsage::Gpu,
                limits
                    .min_uniform_buffer_offset_alignment
                    .into_nonzero()
                    .unwrap(),
                false,
                128,
                cstr::cstr!("Uniform buffer arena").to_owned(),
            ),
            scratch: abs::buffer::BufferArena::new(
                vk::BufferCreateInfo::builder()
                    .size(64 * 128 * 1024)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                abs::memory::MemoryUsage::CpuToGpu,
                nonzero!(1u64),
                true,
                128,
                cstr::cstr!("Scratch buffer arena").to_owned(),
            ),
        }
    }

    pub unsafe fn destroy(
        self,
        device: &ash::Device,
        memory: &mut GpuMemory,
    ) -> anyhow::Result<()> {
        self.vertex.destroy(device, memory)?;
        self.index.destroy(device, memory)?;
        self.uniform.destroy(device, memory)?;
        self.scratch.destroy(device, memory)?;
        Ok(())
    }
}
