mod buddy;
mod buffer;
mod cx;
mod image;
mod memory;
mod util;

use std::time::Duration;

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

    fn redraw(&mut self) -> anyhow::Result<()> {
        unsafe {
            self.cx.device.wait_for_fences(
                &[self.cx.render_fence],
                true,
                Duration::from_secs(1).as_nanos() as u64,
            )?;

            self.cx.device.reset_fences(&[self.cx.render_fence])?;

            let (swapchain_img_index, _is_suboptimal) =
                self.cx.swapchain_loader.acquire_next_image(
                    self.cx.swapchain,
                    Duration::from_secs(1).as_nanos() as u64,
                    self.cx.present_semaphore,
                    ash::vk::Fence::null(),
                )?;

            self.cx.device.reset_command_buffer(
                self.cx.render_cmds,
                ash::vk::CommandBufferResetFlags::empty(),
            )?;

            self.cx.device.begin_command_buffer(
                self.cx.render_cmds,
                &ash::vk::CommandBufferBeginInfo::builder()
                    .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            self.cx.device.cmd_begin_render_pass(
                self.cx.render_cmds,
                &ash::vk::RenderPassBeginInfo::builder()
                    .clear_values(&[ash::vk::ClearValue {
                        color: ash::vk::ClearColorValue {
                            float32: [0., 0., 0., 1.],
                        },
                    }])
                    .render_area(
                        ash::vk::Rect2D::builder()
                            .extent(ash::vk::Extent2D {
                                width: self.cx.width,
                                height: self.cx.height,
                            })
                            .build(),
                    )
                    .render_pass(self.cx.renderpass)
                    .framebuffer(self.cx.framebuffers[swapchain_img_index as usize]),
                ash::vk::SubpassContents::INLINE,
            );

            self.cx.device.cmd_bind_pipeline(
                self.cx.render_cmds,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.cx.pipeline,
            );
            self.cx.device.cmd_draw(self.cx.render_cmds, 3, 1, 0, 0);

            self.cx.device.cmd_end_render_pass(self.cx.render_cmds);
            self.cx.device.end_command_buffer(self.cx.render_cmds)?;

            let wait_stage = &[ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let wait_semaphores = &[self.cx.present_semaphore];
            let signal_semaphores = &[self.cx.render_semaphore];
            let cmd_buffers = &[self.cx.render_cmds];
            let submit = ash::vk::SubmitInfo::builder()
                .wait_dst_stage_mask(wait_stage)
                .signal_semaphores(signal_semaphores)
                .wait_semaphores(wait_semaphores)
                .command_buffers(cmd_buffers)
                .build();

            self.cx
                .device
                .queue_submit(self.cx.render_queue.0, &[submit], self.cx.render_fence)?;

            self.cx.swapchain_loader.queue_present(
                self.cx.render_queue.0,
                &ash::vk::PresentInfoKHR::builder()
                    .swapchains(&[self.cx.swapchain])
                    .wait_semaphores(&[self.cx.render_semaphore])
                    .image_indices(&[swapchain_img_index]),
            )?;

            self.cx.frame += 1;
        }

        Ok(())
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
