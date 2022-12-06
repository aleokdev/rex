mod renderer;

use crate::renderer::Cx;
use std::time::Duration;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

struct App {
    pub cx: Cx,
}

impl App {
    fn new(event_loop: &EventLoop<()>, width: u32, height: u32) -> anyhow::Result<Self> {
        let cx = Cx::new(event_loop, width, height)?;
        Ok(App { cx })
    }

    fn redraw(&mut self) -> anyhow::Result<()> {
        unsafe { self.cx.draw() }
    }
}

pub fn run(width: u32, height: u32) -> anyhow::Result<()> {
    let event_loop = EventLoop::new();
    let mut app = App::new(&event_loop, width, height)?;

    event_loop.run(move |event, _target, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(new_size) => unsafe {
                    app.cx.device.wait_for_fences(
                        &[app.cx.render_queue_fence],
                        true,
                        Duration::from_secs(1).as_nanos() as u64,
                    );
                    app.cx
                        .recreate_swapchain(new_size.width, new_size.height)
                        .unwrap();
                },
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            Event::MainEventsCleared => app.cx.window.request_redraw(),
            Event::RedrawRequested(_) => {
                app.redraw().unwrap();
            }
            _ => {}
        }
    })
}
