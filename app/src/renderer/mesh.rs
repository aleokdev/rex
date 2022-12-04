use ash::vk;

use super::{cx, memory::MemoryUsage};

pub struct Vertex {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
}

pub struct CpuMesh {
    pub vertices: Vec<Vertex>,
}

impl CpuMesh {
    pub unsafe fn upload(&self, cx: &mut cx::Cx) -> anyhow::Result<super::buffer::Buffer> {
        let buffer = cx.memory.allocate_scratch_buffer(
            &vk::BufferCreateInfo::builder()
                .size((self.vertices.len() * std::mem::size_of::<Vertex>()) as u64)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER),
            MemoryUsage::CpuToGpu,
            true,
        )?;
    }
}
