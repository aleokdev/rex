use std::{
    cell::Cell,
    ops::{Deref, DerefMut},
    sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use once_cell::sync::OnceCell;

use crate::abs::memory::GpuMemory;

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

pub struct Memory(GpuMemory);

impl Deref for Memory {
    type Target = GpuMemory;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Memory {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// YES; THIS WILL RESULT IN UNDEFINED BEHAVIOR IF USED FROM MULTIPLE THREADS
// HACK TODO FIXME BAD BAD BAD
unsafe impl Sync for Memory {}
unsafe impl Send for Memory {}

/// The memory used for destroying any object.
///
/// I swear it's not that bad, C does this with malloc and free. Don't blame me.
static MEMORY: OnceCell<RwLock<Memory>> = OnceCell::new();

pub fn get_memory<'m>() -> RwLockReadGuard<'m, Memory> {
    MEMORY
        .get()
        .expect("memory global should be set before usage")
        .read()
        .unwrap()
}

pub fn set_memory(memory: GpuMemory) {
    MEMORY.set(RwLock::new(Memory(memory))).unwrap_or_else(|_| {
        panic!("memory should not have been set before when calling set_memory")
    })
}
