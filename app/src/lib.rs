mod renderer;
use renderer::{
    abs,
    render::{self, Renderer},
    world::{Camera, World},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

struct App {
    pub cx: abs::Cx,
    pub renderer: render::Renderer,
    pub world: World,
}

impl App {
    unsafe fn new(event_loop: &EventLoop<()>, width: u32, height: u32) -> anyhow::Result<Self> {
        let mut cx = abs::Cx::new(event_loop, width, height)?;
        let renderer = render::Renderer::new(&mut cx)?;
        let world = World {
            camera: Camera::new(&cx),
            cube: glam::Vec3::ZERO,
        };

        Ok(App {
            cx,
            renderer,
            world,
        })
    }

    unsafe fn redraw(&mut self) -> anyhow::Result<()> {
        self.renderer.draw(&mut self.cx, &self.world)
    }
}

pub fn run(width: u32, height: u32) -> anyhow::Result<()> {
    let event_loop = EventLoop::new();
    let mut application = Some(unsafe { App::new(&event_loop, width, height)? });

    let mut forward = 0.;
    let mut right = 0.;
    let mut last_mouse = glam::Vec2::ZERO;
    event_loop.run(move |event, _target, control_flow| {
        let Some(app) = application.as_mut() else { return };
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(new_size) if false => unsafe {
                    app.cx.device.device_wait_idle().unwrap();
                    app.cx
                        .recreate_swapchain(new_size.width, new_size.height)
                        .unwrap();
                    app.renderer
                        .resize(&mut app.cx, new_size.width, new_size.height);
                },
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    match (input.virtual_keycode, input.state) {
                        (
                            Some(winit::event::VirtualKeyCode::W),
                            winit::event::ElementState::Pressed,
                        ) => forward += 1.,
                        (
                            Some(winit::event::VirtualKeyCode::W),
                            winit::event::ElementState::Released,
                        ) => forward -= 1.,
                        (
                            Some(winit::event::VirtualKeyCode::S),
                            winit::event::ElementState::Pressed,
                        ) => forward -= 1.,
                        (
                            Some(winit::event::VirtualKeyCode::S),
                            winit::event::ElementState::Released,
                        ) => forward += 1.,
                        (
                            Some(winit::event::VirtualKeyCode::D),
                            winit::event::ElementState::Pressed,
                        ) => right += 1.,
                        (
                            Some(winit::event::VirtualKeyCode::D),
                            winit::event::ElementState::Released,
                        ) => right -= 1.,
                        (
                            Some(winit::event::VirtualKeyCode::A),
                            winit::event::ElementState::Pressed,
                        ) => right -= 1.,
                        (
                            Some(winit::event::VirtualKeyCode::A),
                            winit::event::ElementState::Released,
                        ) => right += 1.,
                        _ => {}
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let pos = glam::Vec2::new(position.x as f32, position.y as f32);
                    if last_mouse == glam::Vec2::ZERO {
                        last_mouse = pos;
                    }
                    let delta = pos - last_mouse;
                    last_mouse = pos;
                    app.world.camera.look(delta.x, delta.y);
                }
                _ => {}
            },
            Event::MainEventsCleared => app.cx.window.request_redraw(),
            Event::RedrawRequested(_) => unsafe {
                app.world.camera.move_forward(forward);
                app.world.camera.move_right(right);
                app.redraw().unwrap();
            },
            Event::LoopDestroyed => unsafe {
                let mut app = application.take().unwrap();
                app.renderer.destroy(&mut app.cx).unwrap();
            },
            _ => {}
        }
    })
}
