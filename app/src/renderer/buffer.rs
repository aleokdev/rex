use super::memory::GpuAllocation;
use ash::vk;

pub struct Buffer {
    pub raw: vk::Buffer,
    pub allocation: GpuAllocation,
}
