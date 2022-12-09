use super::{
    buffer::BufferSlice,
    memory::{GpuMemory, MemoryUsage},
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

pub struct GpuMesh {
    pub vertices: BufferSlice,
    pub indices: BufferSlice,
}

impl GpuMesh {
    pub unsafe fn upload(
        &mut self,
        cx: &mut Cx,
        cmd: vk::CommandBuffer,
        vertices: &[GpuVertex],
        indices: &[u32],
        scratch_memory: &mut GpuMemory,
    ) -> anyhow::Result<()> {
        let vertex_staging = scratch_memory.allocate_scratch_buffer(
            vk::BufferCreateInfo::builder()
                .size(vertices.len() as u64 * std::mem::size_of::<GpuVertex>() as u64)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .build(),
            MemoryUsage::CpuToGpu,
            true,
        )?;

        std::slice::from_raw_parts_mut(
            vertex_staging.allocation.mapped as *mut GpuVertex,
            vertices.len(),
        )
        .copy_from_slice(vertices);

        let index_staging = scratch_memory.allocate_scratch_buffer(
            vk::BufferCreateInfo::builder()
                .size(indices.len() as u64 * 4)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .build(),
            MemoryUsage::CpuToGpu,
            true,
        )?;

        std::slice::from_raw_parts_mut(index_staging.allocation.mapped as *mut u32, indices.len())
            .copy_from_slice(indices);

        cx.device.cmd_copy_buffer(
            cmd,
            vertex_staging.raw,
            self.vertices.buffer,
            &[vk::BufferCopy::builder()
                .src_offset(0)
                .size(vertex_staging.info.size)
                .dst_offset(self.vertices.offset)
                .build()],
        );

        cx.device.cmd_copy_buffer(
            cmd,
            index_staging.raw,
            self.indices.buffer,
            &[vk::BufferCopy::builder()
                .src_offset(0)
                .size(index_staging.info.size)
                .dst_offset(self.indices.offset)
                .build()],
        );

        Ok(())
    }
}
