mod renderer;

use renderer::{abs, render};

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

struct App {
    pub cx: abs::Cx,
    pub renderer: render::Renderer,
}

impl App {
    unsafe fn new(event_loop: &EventLoop<()>, width: u32, height: u32) -> anyhow::Result<Self> {
        let mut cx = abs::Cx::new(event_loop, width, height)?;
        let renderer = render::Renderer::new(&mut cx)?;
        Ok(App { cx, renderer })
    }

    unsafe fn redraw(&mut self) -> anyhow::Result<()> {
        self.renderer.draw(&mut self.cx)
    }
}

pub fn run(width: u32, height: u32) -> anyhow::Result<()> {
    let event_loop = EventLoop::new();
    let mut app = unsafe { App::new(&event_loop, width, height)? };

    event_loop.run(move |event, _target, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(new_size) if false => unsafe {
                    app.cx.device.device_wait_idle().unwrap();
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
            Event::RedrawRequested(_) => unsafe {
                app.redraw().unwrap();
            },
            Event::LoopDestroyed => unsafe {
                app.renderer.destroy(&mut app.cx).unwrap();
            },
            _ => {}
        }
    })
}
