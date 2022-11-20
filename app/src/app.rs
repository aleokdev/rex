use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

use crate::cx::Cx;

struct App {
    pub cx: Cx,
}

impl App {
    fn new(event_loop: &EventLoop<()>, width: u32, height: u32) -> anyhow::Result<Self> {
        let cx = Cx::new(event_loop, width, height)?;
        Ok(App { cx })
    }

    fn redraw(&mut self) {}
}

pub fn run(width: u32, height: u32) -> anyhow::Result<()> {
    let event_loop = EventLoop::new();
    let mut app = App::new(&event_loop, width, height)?;

    event_loop.run(move |event, _target, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(new_size) => unsafe {
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
                app.redraw();
            }
            _ => {}
        }
    })
}
