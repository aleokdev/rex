use anyhow::Result;
use futures::executor::block_on;
use phobos::{pool::ResourcePool, prelude::*};
use std::fs;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::bail;
use glam::{Mat4, Vec3};
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

struct Resources {
    pub offscreen: Image,
    pub offscreen_view: ImageView,
    pub sampler: Sampler,
    pub vertex_buffer: Buffer,
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
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            .build(&event_loop)?;
        Ok(Self { event_loop, window })
    }
}

struct App {
    resources: Resources,
}

impl App {
    fn new(mut ctx: Context) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        // create some pipelines
        // First, we need to load shaders
        let vtx_code = load_spirv_file(Path::new("data/vert.spv"));
        let frag_code = load_spirv_file(Path::new("data/frag.spv"));

        let vertex = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::VERTEX, vtx_code);
        let fragment = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, frag_code);

        // Now we can start using the pipeline builder to create our full pipeline.
        let pci = PipelineBuilder::new("sample".to_string())
            .vertex_input(0, vk::VertexInputRate::VERTEX)
            .vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT)?
            .vertex_attribute(0, 1, vk::Format::R32G32_SFLOAT)?
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .blend_attachment_none()
            .cull_mask(vk::CullModeFlags::NONE)
            .attach_shader(vertex.clone())
            .attach_shader(fragment)
            .build();

        // Store the pipeline in the pipeline cache
        ctx.pool.pipelines.create_named_pipeline(pci)?;

        let frag_code = load_spirv_file(Path::new("data/blue.spv"));
        let fragment = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, frag_code);

        let pci = PipelineBuilder::new("offscreen".to_string())
            .vertex_input(0, vk::VertexInputRate::VERTEX)
            .vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT)?
            .vertex_attribute(0, 1, vk::Format::R32G32_SFLOAT)?
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .blend_attachment_none()
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
        let data: Vec<f32> = vec![
            -1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ];

        let resources = Resources {
            offscreen_view: image.view(vk::ImageAspectFlags::COLOR)?,
            offscreen: image,
            sampler: Sampler::default(ctx.device.clone())?,
            vertex_buffer: staged_buffer_upload(
                ctx.clone(),
                data.as_slice(),
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?,
        };

        Ok(Self { resources })
    }

    fn frame(&mut self, ctx: Context, ifc: InFlightContext) -> anyhow::Result<SubmitBatch<All>> {
        // Define a virtual resource pointing to the swapchain
        let swap_resource = image!("swapchain");
        let offscreen = image!("offscreen");

        let vertices: Vec<f32> = vec![
            -1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ];

        // Define a render graph with one pass that clears the swapchain image
        let graph = PassGraph::new();

        let mut pool = LocalPool::new(ctx.pool.clone())?;

        // Render pass that renders to an offscreen attachment
        let offscreen_pass = PassBuilder::render("offscreen")
            .color([1.0, 0.0, 0.0, 1.0])
            .clear_color_attachment(&offscreen, ClearColor::Float([0.0, 0.0, 0.0, 0.0]))?
            .execute_fn(|mut cmd, ifc, _bindings, _| {
                // Our pass will render a fullscreen quad that 'clears' the screen, just so we can test pipeline creation
                let mut buffer = ifc.allocate_scratch_vbo(
                    (vertices.len() * std::mem::size_of::<f32>()) as vk::DeviceSize,
                )?;
                let slice = buffer.mapped_slice::<f32>()?;
                slice.copy_from_slice(vertices.as_slice());
                cmd = cmd
                    .bind_vertex_buffer(0, &buffer)
                    .bind_graphics_pipeline("offscreen")?
                    .full_viewport_scissor()
                    .draw(6, 1, 0, 0)?;
                Ok(cmd)
            })
            .build();

        // Render pass that samples the offscreen attachment, and possibly does some postprocessing to it
        let sample_pass = PassBuilder::render(String::from("sample"))
            .color([0.0, 1.0, 0.0, 1.0])
            .clear_color_attachment(&swap_resource, ClearColor::Float([0.0, 0.0, 0.0, 0.0]))?
            .sample_image(
                offscreen_pass.output(&offscreen).unwrap(),
                PipelineStage::FRAGMENT_SHADER,
            )
            .execute_fn(|cmd, _ifc, bindings, _| {
                cmd.full_viewport_scissor()
                    .bind_graphics_pipeline("sample")?
                    .resolve_and_bind_sampled_image(
                        0,
                        0,
                        &offscreen,
                        &self.resources.sampler,
                        bindings,
                    )?
                    .draw(6, 1, 0, 0)
            })
            .build();
        // Add another pass to handle presentation to the screen
        let present_pass = PassBuilder::present(
            "present",
            // This pass uses the output from the clear pass on the swap resource as its input
            sample_pass.output(&swap_resource).unwrap(),
        );
        let mut graph = graph
            .add_pass(offscreen_pass)?
            .add_pass(sample_pass)?
            .add_pass(present_pass)?
            // Build the graph, now we can bind physical resources and use it.
            .build()?;

        let mut bindings = PhysicalResourceBindings::new();
        bindings.bind_image("swapchain", &ifc.swapchain_image);
        bindings.bind_image("offscreen", &self.resources.offscreen_view);
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

    fn handle_event(&mut self, _event: &Event<()>) -> Result<()> {
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
            .scratch_size(1 * 1024u64) // 1 KiB scratch memory per buffer type per frame
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
                app.frame(ctx, ifc)
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
        let app = App::new(self.make_context()).unwrap();
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
