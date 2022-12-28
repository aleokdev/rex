use super::{
    allocators::{self, BuddyAllocation},
    buffer::BufferSlice,
    memory::{cmd_stage, GpuMemory},
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

pub struct VertexDescription<const BINDINGS: usize, const DESCRIPTIONS: usize> {
    bindings: [vk::VertexInputBindingDescription; BINDINGS],
    descriptions: [vk::VertexInputAttributeDescription; DESCRIPTIONS],
}

impl<const BINDINGS: usize, const DESCRIPTIONS: usize> VertexDescription<BINDINGS, DESCRIPTIONS> {
    pub fn raw<'s>(&'s self) -> vk::PipelineVertexInputStateCreateInfoBuilder<'s> {
        vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&self.bindings)
            .vertex_attribute_descriptions(&self.descriptions)
    }
}

impl GpuVertex {
    pub fn description<'a>() -> VertexDescription<1, 2> {
        let bindings = [*vk::VertexInputBindingDescription::builder()
            .binding(0)
            .input_rate(vk::VertexInputRate::VERTEX)
            .stride(std::mem::size_of::<GpuVertex>() as u32)];
        let descriptions = [
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(0),
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(12),
        ];
        VertexDescription {
            bindings,
            descriptions,
        }
    }
}

pub struct GpuMesh<Allocation: allocators::Allocation = BuddyAllocation> {
    pub vertices: BufferSlice<Allocation>,
    pub indices: BufferSlice<Allocation>,
    pub vertex_count: u32,
}

impl<Allocation: allocators::Allocation> GpuMesh<Allocation> {
    pub unsafe fn upload(
        &mut self,
        cx: &mut Cx,
        scratch_memory: &mut GpuMemory,
        cmd: vk::CommandBuffer,
        vertices: &[GpuVertex],
        indices: &[u32],
    ) -> anyhow::Result<()> {
        cmd_stage(&cx.device, scratch_memory, cmd, vertices, &self.vertices)?.name(
            cx.device.handle(),
            &cx.debug_utils_loader,
            cstr::cstr!("Mesh Vertex Scratch Buffer"),
        )?;
        cmd_stage(&cx.device, scratch_memory, cmd, indices, &self.indices)?.name(
            cx.device.handle(),
            &cx.debug_utils_loader,
            cstr::cstr!("Mesh Index Scratch Buffer"),
        )?;
        Ok(())
    }
}
