use super::{
    buffer::{Buffer, BufferSlice, OwnedBufferSlice},
    memory::{stage, GpuMemory},
    Cx,
};
use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

pub struct GpuMesh<'b> {
    pub vertices: BufferSlice<'b>,
    pub indices: BufferSlice<'b>,
}

impl<'b> GpuMesh<'b> {
    pub unsafe fn upload(
        &mut self,
        cx: std::sync::Arc<Cx>,
        scratch_memory: &mut GpuMemory,
        cmd: vk::CommandBuffer,
        vertices: &[GpuVertex],
        indices: &[u32],
    ) -> anyhow::Result<()> {
        stage(cx, scratch_memory, cmd, vertices, self.vertices)?;
        stage(cx, scratch_memory, cmd, indices, self.indices)?;
        Ok(())
    }
}
