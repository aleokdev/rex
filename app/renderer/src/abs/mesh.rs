use std::num::NonZeroU64;

use crate::device::get_device;

use super::{
    buffer::{Buffer, BufferSlice},
    memory::{cmd_stage, GpuMemory},
    Cx,
};
use ash::vk;
use nonzero_ext::nonzero;
use space_alloc::{linear::LinearAllocation, BuddyAllocation};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
    pub color: glam::Vec3,
    pub uv: glam::Vec2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub uv: [f32; 2],
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
    pub fn description<'a>() -> VertexDescription<1, 4> {
        let bindings = [*vk::VertexInputBindingDescription::builder()
            .binding(0)
            .input_rate(vk::VertexInputRate::VERTEX)
            .stride(std::mem::size_of::<GpuVertex>() as u32)];
        let descriptions = [
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(4 * 3 * 1),
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(4 * 3 * 2),
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(3)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(4 * 3 * 3),
        ];
        VertexDescription {
            bindings,
            descriptions,
        }
    }

    /// The buffer alignment required for the GPU to read attributes of this vertex.
    pub const fn buffer_alignment_required() -> NonZeroU64 {
        // From https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap22.html#fxvertex-input-extraction:
        // If format is a packed format, attribAddress must be a multiple of the size in
        // bytes of the whole attribute data type as described in Packed Formats. Otherwise,
        // attribAddress must be a multiple of the size in bytes of the component type
        // indicated by format (see Formats).
        // In our case it's unpacked and the component type size in bytes is 4.
        nonzero!(4u64)
    }
}

#[repr(transparent)]
pub struct GpuIndex(pub u32);

impl GpuIndex {
    /// The buffer alignment required for the GPU to read this value.
    pub const fn buffer_alignment_required() -> NonZeroU64 {
        // The sum of offset and the address of the range of VkDeviceMemory object that is backing
        // buffer, must be a multiple of the type indicated by indexType.
        nonzero!(std::mem::size_of::<Self>() as u64)
    }

    pub const fn index_type() -> vk::IndexType {
        vk::IndexType::UINT32
    }
}

pub struct GpuMesh {
    pub vertices: BufferSlice,
    pub indices: BufferSlice,
    pub vertex_count: u32,
}

impl GpuMesh {
    pub unsafe fn upload(
        &mut self,
        cx: &mut Cx,
        scratch_memory: &mut GpuMemory,
        cmd: vk::CommandBuffer,
        vertices: &[GpuVertex],
        indices: &[u32],
    ) -> anyhow::Result<(Buffer<LinearAllocation>, Buffer<LinearAllocation>)> {
        let device = get_device();
        let buf1 = cmd_stage(&device, scratch_memory, cmd, vertices, &self.vertices)?;
        buf1.name(cstr::cstr!("Mesh Vertex Scratch Buffer"))?;
        let buf2 = cmd_stage(&device, scratch_memory, cmd, indices, &self.indices)?;
        buf2.name(cstr::cstr!("Mesh Index Scratch Buffer"))?;
        Ok((buf1, buf2))
    }
}

#[derive(Clone, Copy)]
pub struct GpuMeshHandle(pub(crate) usize);

#[derive(Default, Clone)]
pub struct CpuMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}
