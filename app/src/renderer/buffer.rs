use super::{cx::Cx, memory::GpuAllocation};
use ash::vk;

pub struct Buffer {
    pub raw: vk::Buffer,
    pub allocation: GpuAllocation,
}

impl Buffer {
    pub unsafe fn destroy(self, cx: &mut Cx) -> anyhow::Result<()> {
        cx.memory.free_buffer(self.allocation)?;
        cx.device.destroy_buffer(self.raw, None);
        Ok(())
    }
}
