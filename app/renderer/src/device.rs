use std::sync::Arc;
use std::{
    cell::Cell,
    ops::{Deref, DerefMut},
    sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use ash::extensions::ext::DebugUtils;
use once_cell::sync::OnceCell;

use crate::abs::memory::GpuMemory;

// DECLARATION ORDER IS IMPORTANT! It determines drop order. Be careful.

/// The memory used for destroying any object.
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

struct Device(ash::Device);

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.0.destroy_device(None);
        }
    }
}

/// The device used for destroying any object.
static DEVICE: OnceCell<Device> = OnceCell::new();

pub fn get_device<'d>() -> &'d ash::Device {
    DEVICE
        .get()
        .expect("device global should be set before usage")
}

pub fn set_device(device: ash::Device) {
    DEVICE.set(Device(device)).unwrap_or_else(|_| {
        panic!("device should not have been set before when calling set_device")
    })
}

static DEBUG_UTILS: OnceCell<DebugUtils> = OnceCell::new();

pub fn get_debug_utils<'i>() -> &'i DebugUtils {
    DEBUG_UTILS
        .get()
        .expect("debug utils global should be set before usage")
}

pub fn set_debug_utils(debug_utils: DebugUtils) {
    DEBUG_UTILS.set(debug_utils).unwrap_or_else(|_| {
        panic!("debug_utils should not have been set before when calling set_debug_utils")
    })
}

struct Instance(ash::Instance);

impl Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.0.destroy_instance(None);
        }
    }
}

/// The VkInstance used. We only have a single one per application.
static INSTANCE: OnceCell<Instance> = OnceCell::new();

pub fn get_instance<'i>() -> &'i ash::Instance {
    INSTANCE
        .get()
        .expect("instance global should be set before usage")
}

pub fn set_instance(instance: ash::Instance) {
    INSTANCE.set(Instance(instance)).unwrap_or_else(|_| {
        panic!("instance should not have been set before when calling set_instance")
    })
}
