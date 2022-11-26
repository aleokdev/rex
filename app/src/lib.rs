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
            // - Wait for the render queue fence:
            //      We don't want to submit to the queue while it is busy!
            // - Reset it:
            //      Now that we've finished waiting, the queue is free again, we can reset the
            //      fence.
            // - Obtain the next image we should be drawing onto:
            //      Since we're using a swapchain, the driver should tell us which image of the
            //      swapchain we should be drawing onto; We can't draw directly onto the window.
            // - Reset the command buffer:
            //      We used it last frame (unless this is the first frame, in which case it is
            //      already reset), now that the queue submit is finished we can safely reset it.
            // - Record commands to the buffer:
            //      We now are ready to tell the GPU what to do.
            //      - Begin a render pass:
            //          We clear the whole frame with black.
            //      - Bind our pipeline:
            //          We tell the GPU to configure itself for what's coming...
            //      - Draw:
            //          We draw our mesh having set the pipeline first.
            //      - End the render pass
            // - Submit the command buffer to the queue:
            //      We set it to take place after the image acquisition has been finalized.
            //      This operation will take a while, so we set it to open our render queue fence
            //      once it is complete.
            // - Present the frame drawn:
            //      We adjust this operation to take place after the submission has finished.
            //
            // And thus, our timeline will look something like this:
            // [        CPU        ][        GPU        ]
            // [ Setup GPU work    -> Wait for work     ]
            // [ Wait for fence    ][ Acquire image #1  ]
            // [ Wait for fence    ][ Execute commands  ]
            // [ Wait for fence    <- Signal fence      ]
            // [ Setup GPU work    -> Present image #1  ]
            // [ Wait for fence    ][ Acquire image #2  ]
            self.cx.device.wait_for_fences(
                &[self.cx.render_queue_fence],
                true,
                Duration::from_secs(1).as_nanos() as u64,
            )?;

            self.cx.device.reset_fences(&[self.cx.render_queue_fence])?;

            let (swapchain_img_index, _is_suboptimal) =
                self.cx.swapchain_loader.acquire_next_image(
                    self.cx.swapchain,
                    Duration::from_secs(1).as_nanos() as u64,
                    self.cx.acquire_semaphore,
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
            let wait_semaphores = &[self.cx.acquire_semaphore];
            let signal_semaphores = &[self.cx.render_semaphore];
            let cmd_buffers = &[self.cx.render_cmds];
            let submit = ash::vk::SubmitInfo::builder()
                .wait_dst_stage_mask(wait_stage)
                .wait_semaphores(wait_semaphores)
                .signal_semaphores(signal_semaphores)
                .command_buffers(cmd_buffers)
                .build();

            self.cx.device.queue_submit(
                self.cx.render_queue.0,
                &[submit],
                self.cx.render_queue_fence,
            )?;

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
