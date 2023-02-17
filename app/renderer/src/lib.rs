pub mod abs;
mod camera;
mod data;
mod device;
mod frame;
mod material;
mod object;

pub use camera::Camera;
pub use data::RenderData;
pub(crate) use device::*;
use frame::Frame;
pub use material::Material;
pub use object::RenderObject;

use std::{collections::VecDeque, ffi::CStr};

use crate::abs::{
    mesh::GpuVertex,
    shader::ShaderModule,
    uniforms::{ModelUniform, WorldUniform},
};

use abs::{
    image::{GpuTexture, GpuTextureHandle},
    memory::GpuMemory,
    mesh::{CpuMesh, GpuIndex, GpuMeshHandle, Vertex},
};
use ash::vk::{self};
use nonzero_ext::NonZeroAble;
use space_alloc::{BuddyAllocation, BuddyAllocator, OutOfMemory};

pub struct Renderer {
    pub ds_allocator: abs::descriptor::DescriptorAllocator,
    pub ds_layout_cache: abs::descriptor::DescriptorLayoutCache,
    pub arenas: Arenas,
    pub frame: u64,

    pub vertex_shader: vk::ShaderModule,
    pub fragment_shader: vk::ShaderModule,
    pub wireframe_pipeline: vk::Pipeline,
    pub wireframe_pipeline_layout: vk::PipelineLayout,
    pub flat_lit_pipeline: vk::Pipeline,
    pub flat_lit_pipeline_layout: vk::PipelineLayout,
    pub textured_lit_pipeline: vk::Pipeline,
    pub textured_lit_pipeline_layout: vk::PipelineLayout,

    wireframe: bool,

    pub pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,

    meshes_to_upload: VecDeque<abs::mesh::CpuMesh>,
    // It is important that the `meshes_to_upload` order is respected, as mesh handles' internal ID depend on it!
    meshes: Vec<abs::mesh::GpuMesh>,
    // Same with `images_to_upload`!
    images_to_upload: VecDeque<image::RgbImage>,
    textures: Vec<abs::image::GpuTexture>,
    pub world_uniform: abs::buffer::BufferSlice<BuddyAllocation>,
    pub world_uniform_set: vk::DescriptorSet,
    pub model_uniform: abs::buffer::BufferSlice<BuddyAllocation>,
    pub model_uniform_set: vk::DescriptorSet,

    pub texture_uniform_set_layout: vk::DescriptorSetLayout,
    pub sampler: vk::Sampler,

    pub model_rotation: f32,

    pub frames: [Option<Frame>; Self::FRAME_OVERLAP],
    pub deletion: Vec<Box<dyn FnOnce(&mut abs::Cx)>>,
}

impl Renderer {
    const FRAME_OVERLAP: usize = 2;

    pub unsafe fn new(cx: &mut abs::Cx) -> anyhow::Result<Self> {
        let mut ds_allocator = abs::descriptor::DescriptorAllocator::new(cx.device.clone());
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

        let depth_attachment = vk::AttachmentDescription::builder()
            .format(abs::cx::DEPTH_FORMAT)
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

        let world_uniform = arenas.uniform.suballocate(
            &mut cx.memory,
            cx.device.handle(),
            &cx.debug_utils_loader,
            (std::mem::size_of::<WorldUniform>() as u64)
                .into_nonzero()
                .unwrap(),
        )?;

        let model_uniform = arenas.uniform.suballocate(
            &mut cx.memory,
            cx.device.handle(),
            &cx.debug_utils_loader,
            (std::mem::size_of::<ModelUniform>() as u64)
                .into_nonzero()
                .unwrap(),
        )?;

        let (world_uniform_set, world_uniform_set_layout) =
            abs::descriptor::DescriptorBuilder::new()
                .bind_buffer(
                    0,
                    &[vk::DescriptorBufferInfo::builder()
                        .buffer(world_uniform.buffer.raw)
                        .offset(world_uniform.offset)
                        .range(std::mem::size_of::<WorldUniform>() as u64)
                        .build()],
                    vk::DescriptorType::UNIFORM_BUFFER,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                )
                .build(cx, &mut ds_allocator, &mut ds_layout_cache)?;

        let (model_uniform_set, model_uniform_set_layout) =
            abs::descriptor::DescriptorBuilder::new()
                .bind_buffer(
                    0,
                    &[vk::DescriptorBufferInfo::builder()
                        .buffer(model_uniform.buffer.raw)
                        .offset(model_uniform.offset)
                        .range(std::mem::size_of::<ModelUniform>() as u64)
                        .build()],
                    vk::DescriptorType::UNIFORM_BUFFER,
                    vk::ShaderStageFlags::VERTEX,
                )
                .build(cx, &mut ds_allocator, &mut ds_layout_cache)?;

        let sampler_info = vk::SamplerCreateInfo::builder()
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(false)
            .compare_enable(false)
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR);
        let sampler = cx.device.create_sampler(&sampler_info, None)?;

        let texture_uniform_set_layout = ds_layout_cache.create_descriptor_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                *vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ]),
        );

        let flat_pipeline_layout = cx.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&[world_uniform_set_layout, model_uniform_set_layout]),
            None,
        )?;
        let textured_pipeline_layout = cx.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder().set_layouts(&[
                world_uniform_set_layout,
                model_uniform_set_layout,
                texture_uniform_set_layout,
            ]),
            None,
        )?;

        let vertex_shader =
            ShaderModule::from_spirv_bytes(include_bytes!("../../res/basic.vert.spv"), &cx.device)?
                .name(
                    cx.device.handle(),
                    &cx.debug_utils_loader,
                    cstr::cstr!("Basic Vertex shader"),
                )?
                .0;

        let flat_fragment_shader =
            ShaderModule::from_spirv_bytes(include_bytes!("../../res/flat.frag.spv"), &cx.device)?
                .name(
                    cx.device.handle(),
                    &cx.debug_utils_loader,
                    cstr::cstr!("Basic Fragment shader"),
                )?
                .0;

        let textured_fragment_shader = ShaderModule::from_spirv_bytes(
            include_bytes!("../../res/textured.frag.spv"),
            &cx.device,
        )?
        .name(
            cx.device.handle(),
            &cx.debug_utils_loader,
            cstr::cstr!("Textured Fragment shader"),
        )?
        .0;

        const ENTRY_POINT: &CStr = cstr::cstr!("main");
        let flat_stages = &[
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader)
                .name(ENTRY_POINT)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(flat_fragment_shader)
                .name(ENTRY_POINT)
                .build(),
        ];
        let textured_stages = &[
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader)
                .name(ENTRY_POINT)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(textured_fragment_shader)
                .name(ENTRY_POINT)
                .build(),
        ];

        let wireframe_pipeline =
            create_pipeline(cx, flat_stages, pass, flat_pipeline_layout, true)?;
        let flat_lit_pipeline =
            create_pipeline(cx, flat_stages, pass, flat_pipeline_layout, false)?;
        let textured_lit_pipeline =
            create_pipeline(cx, textured_stages, pass, textured_pipeline_layout, false)?;

        Ok(Renderer {
            ds_allocator,
            ds_layout_cache,
            arenas,
            frame: 0,

            vertex_shader,
            fragment_shader: flat_fragment_shader,
            wireframe_pipeline,
            wireframe_pipeline_layout: flat_pipeline_layout,
            flat_lit_pipeline,
            flat_lit_pipeline_layout: flat_pipeline_layout,
            textured_lit_pipeline,
            textured_lit_pipeline_layout: textured_pipeline_layout,

            wireframe: false,

            pass,
            framebuffers,

            meshes_to_upload: Default::default(),
            meshes: Default::default(),
            images_to_upload: Default::default(),
            textures: Default::default(),
            world_uniform,
            world_uniform_set,
            model_uniform,
            model_uniform_set,
            texture_uniform_set_layout,

            sampler,

            model_rotation: 0.,

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

    pub unsafe fn draw(
        &mut self,
        cx: &mut abs::Cx,
        data: &RenderData,
        _delta: std::time::Duration,
    ) -> anyhow::Result<()> {
        // Get the next frame to draw onto:
        //      We have a few in-flight frames so the GPU doesn't stay still (And we don't need to
        //      wait too much CPU-side for queue submissions to happen).
        let mut frame = self.frames[self.frame as usize % self.frames.len()]
            .take()
            .unwrap();

        // Process the deletion queue for this frame.
        frame.deletion.drain(..).for_each(|f| f(cx));

        // Free all scratch memory, since it's only used for uploading to the GPU.
        frame.allocator.free_scratch();

        // Wait for the frame render queue fence:
        //      We only have command buffer per frame so we need to wait for the previous one before
        //      submitting again, so we can reset it and use it once again.
        cx.device.wait_for_fences(
            &[frame.render_fence],
            true,
            std::time::Duration::from_secs(1).as_nanos() as u64,
        )?;

        // Reset the descriptor set allocator, as all of its used descriptor sets are not being used now.
        frame.ds_allocator.reset()?;

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

        // Update viewport and scissor
        cx.device.cmd_set_viewport(
            frame.cmd,
            0,
            &[*vk::Viewport::builder()
                .width(cx.width as f32)
                .height(cx.height as f32)
                .min_depth(0.)
                .max_depth(1.)],
        );
        cx.device.cmd_set_scissor(
            frame.cmd,
            0,
            &[*vk::Rect2D::builder().extent(vk::Extent2D {
                width: cx.width,
                height: cx.height,
            })],
        );

        // Upload one mesh if available from the queue. We avoid uploading more than one per frame
        // so that we don't impact framerate, but that is a really hacky solution.
        // HACK: Measure frametime instead and upload as much as possible without sacrificing
        // framerate, or even better, use a different upload queue for the job.
        if let Some(mesh) = self.meshes_to_upload.pop_front() {
            match Self::setup_mesh(cx, &mut self.arenas, &mut frame, &mesh) {
                Ok(gpu_mesh) => {
                    self.meshes.push(gpu_mesh);
                }
                Err(err) => {
                    if err.downcast_ref::<OutOfMemory>().is_some() {
                        log::error!(
                            "Could not upload mesh: Out of memory:\nVertex\n{}\n\nIndex\n{}",
                            self.arenas.vertex.info_string(),
                            self.arenas.index.info_string()
                        );
                    } else {
                        return Err(err);
                    }
                }
            }
        }
        if let Some(img) = self.images_to_upload.pop_front() {
            match Self::setup_image(cx, &mut self.arenas, &mut frame, &img) {
                Ok(texture) => {
                    self.textures.push(texture);
                }
                Err(err) => {
                    if err.downcast_ref::<OutOfMemory>().is_some() {
                        log::error!("Could not upload image: Out of memory",);
                    } else {
                        return Err(err);
                    }
                }
            }
        }

        // Upload the uniforms for this frame.
        let world_uniform = WorldUniform {
            proj: data.camera.proj(),
            view: data.camera.view(),
            camera_pos: data.camera.position().extend(0.),
            camera_dir: data.camera.forward().extend(0.),
        };

        // Rotate at 1 sec / turn
        // self.model_rotation += delta.as_secs_f32() * std::f32::consts::TAU;

        let model_uniform = ModelUniform {
            model: glam::Mat4::from_rotation_y(self.model_rotation),
        };

        abs::memory::cmd_stage(
            &cx.device,
            &mut frame.allocator,
            frame.cmd,
            &[world_uniform],
            &self.world_uniform,
        )?
        .name(
            cx.device.handle(),
            &cx.debug_utils_loader,
            cstr::cstr!("Camera Uniform Scratch Buffer"),
        )?;

        abs::memory::cmd_stage(
            &cx.device,
            &mut frame.allocator,
            frame.cmd,
            &[model_uniform],
            &self.model_uniform,
        )?
        .name(
            cx.device.handle(),
            &cx.debug_utils_loader,
            cstr::cstr!("Model Uniform Scratch Buffer"),
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

        for object in data.objects.iter() {
            let Some(mesh) = self.meshes.get(object.mesh_handle.0) else { continue};

            match &object.material {
                Material::FlatLit => {
                    // Bind our pipeline:
                    //      We tell the GPU to configure its layout for what's coming...
                    cx.device.cmd_bind_pipeline(
                        frame.cmd,
                        ash::vk::PipelineBindPoint::GRAPHICS,
                        if self.wireframe {
                            self.wireframe_pipeline
                        } else {
                            self.flat_lit_pipeline
                        },
                    );

                    cx.device.cmd_bind_descriptor_sets(
                        frame.cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.flat_lit_pipeline_layout,
                        0,
                        &[self.world_uniform_set, self.model_uniform_set],
                        &[],
                    );
                }
                Material::TexturedLit { texture } => {
                    let Some(texture) = self.textures.get(texture.0) else{ continue;};
                    // Bind our pipeline:
                    //      We tell the GPU to configure its layout for what's coming...
                    cx.device.cmd_bind_pipeline(
                        frame.cmd,
                        ash::vk::PipelineBindPoint::GRAPHICS,
                        if self.wireframe {
                            self.wireframe_pipeline
                        } else {
                            self.textured_lit_pipeline
                        },
                    );

                    // FIXME: This is allocating one set per frame, but this is not needed. Sets can be updated as long
                    // as the command buffer they are bound to is not being recorded or on the executable state.
                    let texture_uniform_set = frame
                        .ds_allocator
                        .allocate(self.texture_uniform_set_layout)?;
                    cx.device.update_descriptor_sets(
                        &[*vk::WriteDescriptorSet::builder()
                            .dst_binding(0)
                            .dst_set(texture_uniform_set)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(&[*vk::DescriptorImageInfo::builder()
                                .image_view(texture.view)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .sampler(self.sampler)])],
                        &[],
                    );

                    cx.device.cmd_bind_descriptor_sets(
                        frame.cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.textured_lit_pipeline_layout,
                        0,
                        &[
                            self.world_uniform_set,
                            self.model_uniform_set,
                            texture_uniform_set,
                        ],
                        &[],
                    );
                }
            }

            cx.device.cmd_bind_index_buffer(
                frame.cmd,
                mesh.indices.buffer.raw,
                mesh.indices.offset,
                GpuIndex::index_type(),
            );
            cx.device.cmd_bind_vertex_buffers(
                frame.cmd,
                0,
                &[mesh.vertices.buffer.raw],
                &[mesh.vertices.offset],
            );

            // Draw the mesh taking the index buffer into account.
            cx.device
                .cmd_draw_indexed(frame.cmd, mesh.vertex_count, 1, 0, 0, 0);
        }

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

    unsafe fn setup_mesh(
        cx: &mut abs::Cx,
        arenas: &mut Arenas,
        frame: &mut Frame,
        mesh: &CpuMesh,
    ) -> anyhow::Result<abs::mesh::GpuMesh> {
        assert_ne!(mesh.vertices.len(), 0, "mesh must not be empty");
        let gpu_format_vertices = mesh
            .vertices
            .iter()
            .map(|v| GpuVertex {
                position: v.position.to_array(),
                normal: v.normal.to_array(),
                color: v.color.to_array(),
                uv: v.uv.to_array(),
            })
            .collect::<Vec<_>>();
        let vertices_gpu = arenas.vertex.suballocate(
            &mut cx.memory,
            cx.device.handle(),
            &cx.debug_utils_loader,
            ((mesh.vertices.len() * std::mem::size_of::<Vertex>()) as u64)
                .into_nonzero()
                .unwrap(),
        )?;

        let indices_gpu = arenas.index.suballocate(
            &mut cx.memory,
            cx.device.handle(),
            &cx.debug_utils_loader,
            ((mesh.indices.len() * std::mem::size_of::<Vertex>()) as u64)
                .into_nonzero()
                .unwrap(),
        )?;

        let mut gpu_mesh = abs::mesh::GpuMesh {
            indices: indices_gpu,
            vertices: vertices_gpu,
            vertex_count: mesh.indices.len() as u32,
        };

        gpu_mesh.upload(
            cx,
            &mut frame.allocator,
            frame.cmd,
            &gpu_format_vertices,
            &mesh.indices,
        )?;

        Ok(gpu_mesh)
    }

    unsafe fn setup_image(
        cx: &mut abs::Cx,
        arenas: &mut Arenas,
        frame: &mut Frame,
        image: &image::RgbImage,
    ) -> anyhow::Result<abs::image::GpuTexture> {
        let extent = vk::Extent3D {
            width: image.width(),
            height: image.height(),
            depth: 1,
        };
        let info = vk::ImageCreateInfo::builder()
            .extent(extent)
            .array_layers(1)
            .format(vk::Format::B8G8R8A8_SRGB)
            .image_type(vk::ImageType::TYPE_2D)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .tiling(vk::ImageTiling::OPTIMAL)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .mip_levels(1);
        let gpu_img = cx.memory.allocate_image(&info)?;

        let subresource_range = *vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let image_mem_barrier = vk::ImageMemoryBarrier::builder()
            .image(gpu_img.raw)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .subresource_range(subresource_range);
        cx.device.cmd_pipeline_barrier(
            frame.cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[*image_mem_barrier],
        );

        let scratch = cx.memory.allocate_scratch_buffer(
            *vk::BufferCreateInfo::builder()
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .size(image.width() as u64 * image.height() as u64 * 4)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC),
            abs::memory::MemoryUsage::CpuToGpu,
            true,
        )?;
        let pixel_data = image
            .pixels()
            // Convert RGB -> BGRA
            .flat_map(|p| [p.0[2], p.0[1], p.0[0], 0xFF].into_iter())
            .collect::<Vec<u8>>();
        scratch.allocation.write_mapped(&pixel_data)?;

        cx.device.cmd_copy_buffer_to_image(
            frame.cmd,
            scratch.raw,
            gpu_img.raw,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[*vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_extent(extent)
                .image_offset(vk::Offset3D::default())
                .image_subresource(
                    *vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )],
        );

        let image_mem_barrier = vk::ImageMemoryBarrier::builder()
            .image(gpu_img.raw)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .subresource_range(subresource_range);
        cx.device.cmd_pipeline_barrier(
            frame.cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[*image_mem_barrier],
        );

        let view = cx.device.create_image_view(
            &vk::ImageViewCreateInfo::builder()
                .components(vk::ComponentMapping::default())
                .format(vk::Format::B8G8R8A8_SRGB)
                .image(gpu_img.raw)
                .subresource_range(subresource_range)
                .view_type(vk::ImageViewType::TYPE_2D),
            None,
        )?;

        Ok(GpuTexture {
            image: gpu_img,
            view,
        })
    }

    #[must_use = "uploading a mesh does not automatically show it as an object, you must use the handle on an object \
    for it to be useful"]
    pub fn upload_mesh(&mut self, mesh: CpuMesh) -> GpuMeshHandle {
        self.meshes_to_upload.push_back(mesh);
        GpuMeshHandle(self.meshes_to_upload.len() + self.meshes.len() - 1)
    }

    #[must_use = "uploading an image does not automatically show it on screen, you must use the handle on an object \
    for it to be useful"]
    pub fn upload_image(&mut self, image: image::RgbImage) -> GpuTextureHandle {
        self.images_to_upload.push_back(image);
        GpuTextureHandle(self.images_to_upload.len() + self.textures.len() - 1)
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

            drop(frame.ds_allocator);
            drop(frame.allocator);
        });
        for texture in self.textures {
            texture.destroy(&cx.device, &mut cx.memory);
        }

        self.arenas.destroy(&cx.device, &mut cx.memory)?;
        drop(self.ds_layout_cache);
        drop(self.ds_allocator);
        cx.device.destroy_sampler(self.sampler, None);
        cx.device
            .destroy_pipeline_layout(self.textured_lit_pipeline_layout, None);
        cx.device.destroy_pipeline(self.textured_lit_pipeline, None);
        cx.device
            .destroy_pipeline_layout(self.flat_lit_pipeline_layout, None);
        cx.device.destroy_pipeline(self.wireframe_pipeline, None);
        cx.device.destroy_pipeline(self.flat_lit_pipeline, None);
        cx.device.destroy_shader_module(self.vertex_shader, None);
        cx.device.destroy_shader_module(self.fragment_shader, None);
        cx.device.destroy_render_pass(self.pass, None);
        self.framebuffers
            .iter()
            .for_each(|&fb| cx.device.destroy_framebuffer(fb, None));

        Ok(())
    }

    pub fn set_wireframe(&mut self, wireframe: bool) {
        self.wireframe = wireframe;
    }

    pub fn wireframe(&self) -> bool {
        self.wireframe
    }
}

unsafe fn create_pipeline(
    cx: &mut abs::Cx,
    shader_stages: &[vk::PipelineShaderStageCreateInfo; 2],
    pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    wireframe: bool,
) -> Result<vk::Pipeline, anyhow::Error> {
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
        .polygon_mode(if wireframe {
            vk::PolygonMode::LINE
        } else {
            vk::PolygonMode::FILL
        }); // TODO: Cull back facing triangles
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
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
    let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(shader_stages)
        .input_assembly_state(&input_assembly_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&msaa)
        .render_pass(pass)
        .subpass(0)
        .layout(pipeline_layout)
        .color_blend_state(&color_blend_state)
        .depth_stencil_state(&depth_stencil_state)
        .viewport_state(&viewport_state)
        .vertex_input_state(&*vertex_input_state_raw)
        .dynamic_state(&dynamic_state);
    let pipeline = cx
        .device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[*pipeline_info], None)
        .map_err(|(_, e)| e)?[0];
    Ok(pipeline)
}

pub struct Arenas {
    pub vertex: abs::buffer::BufferArena<BuddyAllocator>,
    pub index: abs::buffer::BufferArena<BuddyAllocator>,
    pub uniform: abs::buffer::BufferArena<BuddyAllocator>,
}

impl Arenas {
    pub fn new(limits: &vk::PhysicalDeviceLimits) -> Self {
        Arenas {
            vertex: abs::buffer::BufferArena::new(
                vk::BufferCreateInfo::builder()
                    .size(256 * 1024 * std::mem::size_of::<abs::mesh::GpuVertex>() as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                abs::memory::MemoryUsage::Gpu,
                GpuVertex::buffer_alignment_required(),
                false,
                256 * std::mem::size_of::<abs::mesh::GpuVertex>() as u64,
                cstr::cstr!("Vertex buffer arena").to_owned(),
            ),
            index: abs::buffer::BufferArena::new(
                vk::BufferCreateInfo::builder()
                    .size(2 * 1024 * 1024 * std::mem::size_of::<GpuIndex>() as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                abs::memory::MemoryUsage::Gpu,
                GpuIndex::buffer_alignment_required(),
                false,
                256 * std::mem::size_of::<GpuIndex>() as u64,
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
        Ok(())
    }
}
