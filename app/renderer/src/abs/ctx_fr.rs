use std::{ops::Deref, sync::Arc};

// TODO clean this up

/// The context, but for real.
pub struct CtxFr(Arc<CtxFrInstance>);

impl Deref for CtxFr {
    type Target = CtxFrInstance;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

pub struct CtxFrInstance {
    device: ash::Device,
    memory: GpuMemory,
}

impl CtxFrInstance {
    fn device(&self) -> &ash::Device {
        &self.device
    }

    fn memory(&self) -> &GpuMemory {
        &self.memory
    }
}
