use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use crevice::std140::AsStd140;
use futures::executor::block_on;
use phobos::vk::{AttachmentLoadOp, ClearDepthStencilValue, CompareOp, IndexType};
use phobos::{pool::ResourcePool, prelude::*};
use std::cell::OnceCell;
use std::collections::HashMap;
use std::fs::File;
use std::fs::{self, read_to_string};
use std::io::{Read, Write};
use std::path::Path;

use anyhow::bail;
use glam::{Mat4, Quat, Vec3};
use winit::event::{
    ElementState, Event, MouseButton, MouseScrollDelta, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::{ControlFlow, EventLoop, EventLoopBuilder};
use winit::window::Window;
use winit::window::WindowBuilder;

use phobos::command_buffer::traits::*;
use phobos::graph::pass::ClearColor;
use phobos::pool::LocalPool;
use phobos::prelude::*;
use phobos::sync::domain::All;
use phobos::sync::submit_batch::SubmitBatch;
use phobos::{image, vk};

use egui_winit_phobos::Integration as EguiIntegration;

const SHADER_COMPILER: OnceCell<shaderc::Compiler> = OnceCell::new();

pub fn staged_buffer_upload<T: Copy>(
    mut ctx: Context,
    data: &[T],
    usage: vk::BufferUsageFlags,
) -> Result<Buffer> {
    let staging = Buffer::new(
        ctx.device.clone(),
        &mut ctx.allocator,
        data.len() as u64 * std::mem::size_of::<T>() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryType::CpuToGpu,
    )?;

    let mut staging_view = staging.view_full();
    staging_view.mapped_slice()?.copy_from_slice(data);

    let buffer = Buffer::new_device_local(
        ctx.device,
        &mut ctx.allocator,
        staging.size(),
        vk::BufferUsageFlags::TRANSFER_DST | usage,
    )?;
    let view = buffer.view_full();

    let cmd = ctx
        .exec
        .on_domain::<domain::Transfer>()?
        .copy_buffer(&staging_view, &view)?
        .finish()?;

    ctx.exec.submit(cmd)?.wait()?;
    Ok(buffer)
}

pub fn load_spirv_file(path: &Path) -> Vec<u32> {
    let mut f = File::open(&path).expect("no file found");
    let metadata = fs::metadata(&path).expect("unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    f.read(&mut buffer).expect("buffer overflow");
    let (_, binary, _) = unsafe { buffer.align_to::<u32>() };
    Vec::from(binary)
}

pub fn compile_glsl_file(path: &Path, kind: shaderc::ShaderKind) -> Vec<u32> {
    SHADER_COMPILER
        .get_or_init(|| shaderc::Compiler::new().expect("new shaderc compiler"))
        .compile_into_spirv(
            &read_to_string(path).expect("read file"),
            kind,
            path.file_name().unwrap().to_str().unwrap(),
            "main",
            None,
        )
        .expect("compile glsl")
        .as_binary()
        .to_owned()
}

pub struct VulkanContext {
    pub frame: Option<FrameManager>,
    pub pool: ResourcePool,
    pub exec: ExecutionManager,
    pub allocator: DefaultAllocator,
    pub device: Device,
    pub physical_device: PhysicalDevice,
    pub surface: Option<Surface>,
    pub debug_messenger: DebugMessenger,
    pub instance: Instance,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

struct Mesh {
    pub vertices: Buffer,
    pub indices: Buffer,
    pub index_count: u64,
}

impl Mesh {
    pub fn load_gltf(mut ctx: Context, path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let (doc, bufs, images) = gltf::import(path)?;
        let primitive0 = doc.meshes().next().unwrap().primitives().next().unwrap();
        let mesh_reader = primitive0.reader(|b| Some(&bufs[b.index()].0));
        let vertex_data: Vec<_> = mesh_reader
            .read_positions()
            .unwrap()
            .zip(mesh_reader.read_normals().unwrap())
            .zip(mesh_reader.read_tex_coords(0).unwrap().into_f32())
            .map(|((pos, normal), uv)| GpuVertex { pos, normal, uv })
            .collect();
        let index_data: Vec<_> = mesh_reader.read_indices().unwrap().into_u32().collect();

        let vertices = staged_buffer_upload(
            ctx.clone(),
            &vertex_data,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let indices = staged_buffer_upload(ctx, &index_data, vk::BufferUsageFlags::INDEX_BUFFER)?;

        Ok(Mesh {
            vertices,
            indices,
            index_count: index_data.len() as _,
        })
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct MeshHandle(pub usize);

struct Object {
    pub mesh: MeshHandle,
    pub transform: Mat4,
}

struct Resources {
    pub offscreen: Image,
    pub offscreen_view: ImageView,
    pub depth: Image,
    pub depth_view: ImageView,
    pub ui: Image,
    pub ui_view: ImageView,
    pub sampler: Sampler,
    pub meshes: Vec<Mesh>,
    pub objects: Vec<Object>,
}

impl Resources {
    pub fn push_mesh(&mut self, mesh: Mesh) -> MeshHandle {
        self.meshes.push(mesh);
        MeshHandle(self.meshes.len() - 1)
    }

    pub fn mesh(&self, handle: MeshHandle) -> &Mesh {
        self.meshes.get(handle.0).expect("invalid mesh")
    }
}

#[derive(Clone)]
pub struct Context {
    pub device: Device,
    pub exec: ExecutionManager,
    pub allocator: DefaultAllocator,
    pub pool: ResourcePool,
}

#[derive(Debug)]
pub struct WindowContext {
    pub event_loop: EventLoop<()>,
    pub window: Window,
}

impl WindowContext {
    pub fn new(title: impl Into<String>) -> Result<Self> {
        Self::with_size(title, 800.0, 600.0)
    }

    pub fn with_size(title: impl Into<String>, width: f32, height: f32) -> Result<Self> {
        let event_loop = EventLoopBuilder::new().build();
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(winit::dpi::PhysicalSize::new(width, height))
            .build(&event_loop)?;
        Ok(Self { event_loop, window })
    }
}

struct App {
    resources: Resources,
    egui: EguiIntegration<DefaultAllocator>,
    time: f32,
}

impl App {
    fn new(mut ctx: Context, event_loop: &EventLoop<()>) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        // create some pipelines
        // First, we need to load shaders
        // let vtx_code = load_spirv_file(Path::new("data/vert.spv"));
        // let frag_code = load_spirv_file(Path::new("data/frag.spv"));
        let vtx_code = compile_glsl_file(Path::new("data/cube.vert"), shaderc::ShaderKind::Vertex);
        let frag_code =
            compile_glsl_file(Path::new("data/cube.frag"), shaderc::ShaderKind::Fragment);
        let vertex = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::VERTEX, vtx_code);
        let fragment = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, frag_code);

        let pci = PipelineBuilder::new("offscreen".to_string())
            .vertex_input(0, vk::VertexInputRate::VERTEX)
            .vertex_attribute(0, 0, vk::Format::R32G32B32_SFLOAT)?
            .vertex_attribute(0, 1, vk::Format::R32G32B32_SFLOAT)?
            .vertex_attribute(0, 2, vk::Format::R32G32_SFLOAT)?
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .depth(true, true, false, CompareOp::LESS)
            .blend_attachment_none()
            .cull_mask(vk::CullModeFlags::NONE)
            .attach_shader(vertex)
            .attach_shader(fragment)
            .build();
        ctx.pool.pipelines.create_named_pipeline(pci)?;

        let vtx_code = compile_glsl_file(Path::new("data/comp.vert"), shaderc::ShaderKind::Vertex);
        let frag_code =
            compile_glsl_file(Path::new("data/comp.frag"), shaderc::ShaderKind::Fragment);
        let vertex = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::VERTEX, vtx_code);
        let fragment = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, frag_code);

        let pci = PipelineBuilder::new("composite".to_string())
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .blend_additive_unmasked(
                vk::BlendFactor::SRC_ALPHA,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                vk::BlendFactor::ONE_MINUS_DST_ALPHA,
                vk::BlendFactor::ONE,
            )
            .cull_mask(vk::CullModeFlags::NONE)
            .attach_shader(vertex)
            .attach_shader(fragment)
            .build();
        ctx.pool.pipelines.create_named_pipeline(pci)?;

        // Define some resources we will use for rendering
        let image = Image::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            800,
            600,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::Format::R8G8B8A8_SRGB,
            vk::SampleCountFlags::TYPE_1,
        )?;

        let ui = Image::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            800,
            600,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::Format::R8G8B8A8_SRGB,
            vk::SampleCountFlags::TYPE_1,
        )?;

        let depth = Image::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            800,
            600,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::Format::D32_SFLOAT,
            vk::SampleCountFlags::TYPE_1,
        )?;

        // bleeding edge advanced GLTF mesh importer

        let mut resources = Resources {
            offscreen_view: image.view(vk::ImageAspectFlags::COLOR)?,
            offscreen: image,
            depth_view: depth.view(vk::ImageAspectFlags::DEPTH)?,
            depth,
            ui_view: ui.view(vk::ImageAspectFlags::COLOR)?,
            ui,
            sampler: Sampler::default(ctx.device.clone())?,
            meshes: vec![],
            objects: vec![],
        };

        let cube_mesh = resources.push_mesh(Mesh::load_gltf(ctx.clone(), "data/cube.gltf")?);

        resources.objects.push(Object {
            mesh: cube_mesh,
            transform: Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, Vec3::ZERO),
        });

        let mut fonts = egui::FontDefinitions::empty();
        fonts.font_data.insert(
            "test_font".into(),
            egui::FontData::from_static(include_bytes!("../data/DejaVuSans.ttf")),
        );
        fonts
            .families
            .get_mut(&egui::FontFamily::Proportional)
            .unwrap()
            .insert(0, "test_font".into());

        let egui = EguiIntegration::new(
            800,
            600,
            1.0,
            &event_loop,
            Default::default(),
            Default::default(),
            ctx.device.clone(),
            ctx.allocator.clone(),
            ctx.exec.clone(),
            ctx.pool.pipelines.clone(),
        )?;

        Ok(Self {
            resources,
            egui,
            time: 0.,
        })
    }

    fn frame(
        &mut self,
        ctx: Context,
        ifc: InFlightContext,
        window: &Window,
    ) -> anyhow::Result<SubmitBatch<All>> {
        // Define a virtual resource pointing to the swapchain
        let swap = image!("swapchain");
        let color = image!("color");
        let depth = image!("depth");
        let ui = image!("ui");

        // Define a render graph with one pass that clears the swapchain image
        let graph = PassGraph::new();

        let mut pool = LocalPool::new(ctx.pool.clone())?;

        #[derive(AsStd140)]
        struct CameraUniforms {
            pub vp: mint::ColumnMatrix4<f32>,
        }

        #[derive(AsStd140)]
        struct ModelUniforms {
            pub model: mint::ColumnMatrix4<f32>,
            pub normal: mint::ColumnMatrix4<f32>,
        }

        self.time += 0.01; // dt? whats that

        // Render pass that renders to an offscreen attachment
        let offscreen_pass = PassBuilder::render("offscreen")
            .color([1.0, 0.0, 0.0, 1.0])
            .clear_color_attachment(&color, ClearColor::Float([0.0, 0.0, 0.0, 1.0]))?
            .clear_depth_attachment(
                &depth,
                ClearDepthStencil {
                    depth: 1.,
                    stencil: 0,
                },
            )?
            .execute_fn(|mut cmd, pool, _bindings, _| {
                let model = Mat4::from_rotation_x(self.time);
                let camera_uniforms = CameraUniforms {
                    vp: (Mat4::perspective_rh(60.0f32.to_radians(), 800.0 / 600.0, 0.1, 100.0)
                        * Mat4::look_at_rh(
                            Vec3::new(2.0, -2.0, 2.0),
                            Vec3::ZERO,
                            Vec3::new(0.0, -1.0, 0.0),
                        ))
                    .into(),
                };
                let mut scratch =
                    pool.allocate_scratch_ubo(CameraUniforms::std140_size_static() as _)?;
                scratch
                    .mapped_slice()?
                    .copy_from_slice(camera_uniforms.as_std140().as_bytes());

                // Our pass will render a fullscreen quad that 'clears' the screen, just so we can test pipeline creation
                cmd = cmd
                    .bind_graphics_pipeline("offscreen")?
                    .bind_uniform_buffer(0, 0, &scratch)?
                    .full_viewport_scissor();

                for obj in &self.resources.objects {
                    let model_uniforms = ModelUniforms {
                        model: obj.transform.into(),
                        normal: obj.transform.inverse().transpose().into(),
                    };

                    let mut obj_scratch =
                        pool.allocate_scratch_ubo(ModelUniforms::std140_size_static() as _)?;
                    obj_scratch
                        .mapped_slice()?
                        .copy_from_slice(model_uniforms.as_std140().as_bytes());

                    let mesh = self.resources.mesh(obj.mesh);
                    cmd = cmd
                        .bind_vertex_buffer(0, &mesh.vertices.view_full())
                        .bind_index_buffer(&mesh.indices.view_full(), IndexType::UINT32)
                        .bind_uniform_buffer(1, 0, &obj_scratch)?
                        .draw_indexed(mesh.index_count as _, 1, 0, 0, 0)?;
                }

                Ok(cmd)
            })
            .build();

        self.egui.begin_frame(window);

        egui::Window::new("voices (very Loud)").show(&self.egui.context(), |ui| {
            ui.label("TEST");
            ui.button("clik me");
        });

        let ui_data = self.egui.end_frame(window);

        let egui_pass = self.egui.paint(
            &[],
            &ui,
            AttachmentLoadOp::CLEAR,
            Some(vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            }),
            self.egui.context().tessellate(ui_data.shapes),
            ui_data.textures_delta,
        )?;

        let composite_pass = PassBuilder::render("composite")
            .color([0.0, 1.0, 0.0, 1.0])
            .clear_color_attachment(&swap, ClearColor::Float([0.0, 0.0, 0.0, 1.0]))?
            .sample_image(&color, vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .sample_image(&ui, vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .execute_fn(|mut cmd, pool, bindings, _| {
                cmd = cmd
                    .bind_graphics_pipeline("composite")?
                    .full_viewport_scissor()
                    .resolve_and_bind_sampled_image(
                        0,
                        0,
                        &color,
                        &self.resources.sampler,
                        bindings,
                    )?
                    .draw(3, 1, 0, 0)?
                    .resolve_and_bind_sampled_image(0, 0, &ui, &self.resources.sampler, bindings)?
                    .draw(3, 1, 0, 0)?;
                Ok(cmd)
            })
            .build();

        // Add another pass to handle presentation to the screen
        let present_pass = PassBuilder::present(
            "present",
            // This pass uses the output from the clear pass on the swap resource as its input
            composite_pass.output(&swap).unwrap(),
        );
        let mut graph = graph
            .add_pass(offscreen_pass)?
            .add_pass(egui_pass)?
            .add_pass(composite_pass)?
            .add_pass(present_pass)?
            // Build the graph, now we can bind physical resources and use it.
            .build()?;

        let mut bindings = PhysicalResourceBindings::new();
        bindings.bind_image("swapchain", &ifc.swapchain_image);
        bindings.bind_image("color", &self.resources.offscreen_view);
        bindings.bind_image("depth", &self.resources.depth_view);
        bindings.bind_image("ui", &self.resources.ui_view);
        // create a command buffer capable of executing graphics commands
        let cmd = ctx.exec.on_domain::<All>().unwrap();
        // record render graph to this command buffer
        let cmd = graph
            .record(cmd, &bindings, &mut pool, None, &mut ())?
            .finish()?;
        let mut batch = ctx.exec.start_submit_batch()?;
        batch.submit_for_present(cmd, ifc, pool)?;
        Ok(batch)
    }

    // Implement this for a headless application
    fn run(&mut self, _ctx: Context) -> Result<()> {
        bail!("run() not implemented for headless example app");
    }

    fn handle_event(&mut self, event: &Event<()>) -> Result<()> {
        match event {
            Event::WindowEvent { event, .. } => {
                self.egui.handle_event(&event);
            }
            _ => {}
        }

        Ok(())
    }
}

struct AppRunner {
    vk: VulkanContext,
}

impl AppRunner {
    pub fn new(
        name: impl Into<String>,
        window: Option<&WindowContext>,
        make_settings: impl Fn(AppBuilder<Window>) -> AppSettings<Window>,
    ) -> anyhow::Result<Self> {
        std::env::set_var("RUST_LOG", "trace");
        pretty_env_logger::init();
        let mut settings = AppBuilder::new()
            .version((1, 0, 0))
            .name(name)
            .validation(true)
            .present_mode(vk::PresentModeKHR::MAILBOX)
            .scratch_size(8 * 1024 * 1024u64) // 8 MiB scratch memory per buffer type per frame
            .gpu(GPURequirements {
                dedicated: false,
                min_video_memory: 1 * 1024 * 1024 * 1024, // 1 GiB.
                min_dedicated_video_memory: 1 * 1024 * 1024 * 1024,
                queues: vec![
                    QueueRequest {
                        dedicated: false,
                        queue_type: QueueType::Graphics,
                    },
                    QueueRequest {
                        dedicated: true,
                        queue_type: QueueType::Transfer,
                    },
                    QueueRequest {
                        dedicated: true,
                        queue_type: QueueType::Compute,
                    },
                ],
                ..Default::default()
            });

        match window {
            None => {}
            Some(window) => {
                settings = settings.window(&window.window);
            }
        };
        let settings = make_settings(settings);

        let (instance, physical_device, surface, device, allocator, pool, exec, frame, Some(debug_messenger)) = initialize(&settings, window.is_none())? else {
            panic!("Asked for debug messenger but didnt get one")
        };

        let vk = VulkanContext {
            frame,
            pool,
            exec,
            allocator,
            device,
            physical_device,
            surface,
            debug_messenger,
            instance,
        };

        Ok(Self { vk })
    }

    fn make_context(&self) -> Context {
        Context {
            device: self.vk.device.clone(),
            exec: self.vk.exec.clone(),
            allocator: self.vk.allocator.clone(),
            pool: self.vk.pool.clone(),
        }
    }

    fn frame(&mut self, app: &mut App, window: &Window) -> Result<()> {
        let ctx = self.make_context();
        let frame = self.vk.frame.as_mut().unwrap();
        let surface = self.vk.surface.as_ref().unwrap();
        block_on(
            frame.new_frame(self.vk.exec.clone(), window, surface, |ifc| {
                app.frame(ctx, ifc, window)
            }),
        )?;

        Ok(())
    }

    fn run_windowed(mut self, app: App, window: WindowContext) -> ! {
        let event_loop = window.event_loop;
        let window = window.window;
        let mut app = Some(app);
        event_loop.run(move |event, _, control_flow| {
            // Do not render a frame if Exit control flow is specified, to avoid
            // sync issues.
            if let ControlFlow::ExitWithCode(_) = *control_flow {
                self.vk.device.wait_idle().unwrap();
                return;
            }
            *control_flow = ControlFlow::Poll;

            match &mut app {
                None => {}
                Some(app) => {
                    app.handle_event(&event).unwrap();
                }
            }

            // Note that we want to handle events after processing our current frame, so that
            // requesting an exit doesn't attempt to render another frame, which causes
            // sync issues.
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == window.id() => {
                    *control_flow = ControlFlow::Exit;
                    self.vk.device.wait_idle().unwrap();
                    let app = app.take();
                    match app {
                        None => {}
                        Some(app) => {
                            drop(app);
                        }
                    }
                }
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(_) => match app.as_mut() {
                    None => {}
                    Some(app) => {
                        self.frame(app, &window).unwrap();
                        self.vk.pool.next_frame();
                    }
                },
                _ => (),
            }
        })
    }

    pub fn run(self, window: WindowContext) -> ! {
        let app = App::new(self.make_context(), &window.event_loop).unwrap();
        match window {
            window => self.run_windowed(app, window),
        }
    }
}

fn main() -> Result<()> {
    let window = WindowContext::new("01_basic")?;
    AppRunner::new("example", Some(&window), |s| s.build())?.run(window);
    Ok(())
}
