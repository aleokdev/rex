use std::sync::Arc;
use std::{
    cell::Cell,
    ops::{Deref, DerefMut},
    sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use once_cell::sync::OnceCell;

use crate::abs::memory::GpuMemory;

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

// todo: have a ctxfr instead?
/// The device used for destroying any object.
///
/// Hate me for this, but I think it's better than having one device reference per GPU handle.
static DEVICE: OnceCell<ash::Device> = OnceCell::new();

pub fn get_device<'d>() -> &'d ash::Device {
    DEVICE
        .get()
        .expect("device global should be set before usage")
}

pub fn set_device(device: ash::Device) {
    DEVICE.set(device).unwrap_or_else(|_| {
        panic!("device should not have been set before when calling set_device")
    })
}

/// The memory used for destroying any object.
///
/// I swear it's not that bad, C does this with malloc and free. Don't blame me.
static MEMORY: OnceCell<RwLock<GpuMemory>> = OnceCell::new();

pub fn get_memory<'m>() -> RwLockReadGuard<'m, GpuMemory> {
    MEMORY
        .get()
        .expect("memory global should be set before usage")
        .read()
        .unwrap()
}

pub fn set_memory(memory: GpuMemory) {
    MEMORY.set(RwLock::new(memory)).unwrap_or_else(|_| {
        panic!("memory should not have been set before when calling set_memory")
    })
}
