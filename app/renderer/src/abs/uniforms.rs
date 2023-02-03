#[repr(C)]
#[derive(Clone, Copy)]
pub struct WorldUniform {
    pub proj: glam::Mat4,
    pub view: glam::Mat4,
    pub camera_pos: glam::Vec4,
    pub camera_dir: glam::Vec4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ModelUniform {
    pub model: glam::Mat4,
}
