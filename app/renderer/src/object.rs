use crate::{abs::mesh::GpuMeshHandle, material::Material};

pub struct RenderObject {
    pub mesh_handle: GpuMeshHandle,
    pub material: Material,
    pub transform: glam::Mat4,
}
