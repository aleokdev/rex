use ash::vk;

use crate::get_device;

pub struct Cmd {
    // TODO: Make private
    pub(crate) buffer: vk::CommandBuffer,
    // TODO: Make private?
    pub(crate) deletion_queue: Vec<Box<dyn FnOnce()>>,
}

pub trait QueueExecutable: Sized {
    fn queue<'cmd>(self) -> Cmd;

    fn set_viewport(self, viewport: vk::Viewport) -> SetViewport<Self> {
        SetViewport {
            viewport,
            inner: self,
        }
    }

    fn set_scissor(self, scissor: vk::Rect2D) -> SetScissor<Self> {
        SetScissor {
            scissor,
            inner: self,
        }
    }

    fn queue_try_fn<E, Fout: FnOnce(Cmd) -> Cmd, F: FnOnce() -> Result<Fout, E>>(
        self,
        func: F,
    ) -> Result<QueueFn<Self, Fout>, E> {
        Ok(QueueFn {
            func: func()?,
            inner: self,
        })
    }

    fn queue_fn<F: FnOnce(Cmd) -> Cmd>(self, func: F) -> QueueFn<Self, F> {
        QueueFn { func, inner: self }
    }
}

impl QueueExecutable for Cmd {
    fn queue(self) -> Cmd {
        self
    }
}

pub struct QueueFn<C: QueueExecutable, F: FnOnce(Cmd) -> Cmd> {
    func: F,
    inner: C,
}

impl<C: QueueExecutable, F: FnOnce(Cmd) -> Cmd> QueueExecutable for QueueFn<C, F> {
    fn queue(self) -> Cmd {
        (self.func)(self.inner.queue())
    }
}

pub struct SetViewport<C: QueueExecutable> {
    viewport: vk::Viewport,
    inner: C,
}

impl<C: QueueExecutable> QueueExecutable for SetViewport<C> {
    fn queue(self) -> Cmd {
        let cmd = self.inner.queue();
        unsafe { get_device().cmd_set_viewport(cmd.buffer, 0, &[self.viewport]) };
        cmd
    }
}

pub struct SetScissor<C: QueueExecutable> {
    scissor: vk::Rect2D,
    inner: C,
}

impl<C: QueueExecutable> QueueExecutable for SetScissor<C> {
    fn queue(self) -> Cmd {
        let cmd = self.inner.queue();
        unsafe { get_device().cmd_set_scissor(cmd.buffer, 0, &[self.scissor]) };
        cmd
    }
}
