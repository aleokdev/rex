mod renderer;
mod world;

use glam::Vec3;
use renderer::{
    abs,
    render::{self},
};
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::CursorGrabMode,
};
use world::{Camera, World};

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
            camera: Camera::new(cx.width as f32 / cx.height as f32),
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

    let mut last_time = std::time::Instant::now();
    let mut movement_keys_pressed = [false; 6];
    let mut sprint_key_pressed = false;
    let mut focused = false;
    event_loop.run(move |event, _target, control_flow| {
        let Some(app) = application.as_mut() else { return };
        *control_flow = ControlFlow::Poll;

        let now_time = std::time::Instant::now();
        let delta = now_time - last_time;
        let delta_s = delta.as_secs_f32();
        const SPEED: f32 = 100.;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(new_size) => unsafe {
                    app.cx.device.device_wait_idle().unwrap();
                    app.cx
                        .recreate_swapchain(new_size.width, new_size.height)
                        .unwrap();
                    app.renderer
                        .resize(&mut app.cx, new_size.width, new_size.height)
                        .unwrap();
                },
                WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state,
                            ..
                        },
                    ..
                } => match keycode {
                    winit::event::VirtualKeyCode::W => {
                        movement_keys_pressed[0] = state == winit::event::ElementState::Pressed
                    }
                    winit::event::VirtualKeyCode::S => {
                        movement_keys_pressed[1] = state == winit::event::ElementState::Pressed
                    }
                    winit::event::VirtualKeyCode::D => {
                        movement_keys_pressed[2] = state == winit::event::ElementState::Pressed
                    }
                    winit::event::VirtualKeyCode::A => {
                        movement_keys_pressed[3] = state == winit::event::ElementState::Pressed
                    }
                    winit::event::VirtualKeyCode::Space => {
                        movement_keys_pressed[4] = state == winit::event::ElementState::Pressed
                    }
                    winit::event::VirtualKeyCode::LShift => {
                        movement_keys_pressed[5] = state == winit::event::ElementState::Pressed
                    }
                    winit::event::VirtualKeyCode::LControl => {
                        sprint_key_pressed = state == winit::event::ElementState::Pressed
                    }

                    winit::event::VirtualKeyCode::Escape
                        if state == winit::event::ElementState::Pressed =>
                    {
                        focused = false;
                        app.cx.window.set_cursor_visible(true);
                        drop(app.cx.window.set_cursor_grab(CursorGrabMode::None));
                    }
                    _ => {}
                },
                WindowEvent::CursorMoved { .. } if focused => {
                    let PhysicalSize { width, height } = app.cx.window.inner_size();

                    drop(
                        app.cx
                            .window
                            .set_cursor_position(PhysicalPosition::new(width / 2, height / 2)),
                    );
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    ..
                } => {
                    focused = true;
                    app.cx.window.set_cursor_visible(false);
                    drop(app.cx.window.set_cursor_grab(CursorGrabMode::Confined));
                }
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            Event::DeviceEvent { event, .. } if focused => match event {
                winit::event::DeviceEvent::MouseMotion { delta } => {
                    const LOOK_SENSITIVITY: f32 = 0.001;
                    app.world.camera.look(
                        -delta.0 as f32 * LOOK_SENSITIVITY,
                        -delta.1 as f32 * LOOK_SENSITIVITY,
                    );
                }
                _ => (),
            },
            Event::MainEventsCleared => app.cx.window.request_redraw(),
            Event::RedrawRequested(_) => unsafe {
                let mut movement = Vec3::ZERO;
                movement.z += if movement_keys_pressed[0] { 1. } else { 0. };
                movement.z -= if movement_keys_pressed[1] { 1. } else { 0. };
                movement.x -= if movement_keys_pressed[2] { 1. } else { 0. };
                movement.x += if movement_keys_pressed[3] { 1. } else { 0. };
                movement.y += if movement_keys_pressed[4] { 1. } else { 0. };
                movement.y -= if movement_keys_pressed[5] { 1. } else { 0. };
                if sprint_key_pressed {
                    movement *= 10.;
                }
                log::info!(
                    "{}, forward {}, up {}",
                    app.world.camera.position(),
                    app.world.camera.forward(),
                    app.world.camera.up()
                );
                app.world
                    .camera
                    .move_local_coords(movement * SPEED * delta_s);
                app.redraw().unwrap();
            },
            Event::LoopDestroyed => unsafe {
                let mut app = application.take().unwrap();
                app.renderer.destroy(&mut app.cx).unwrap();
            },
            _ => {}
        }

        last_time = now_time;
    })
}
