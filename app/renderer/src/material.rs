use crate::abs::image::GpuTextureHandle;

pub enum Material {
    FlatLit,
    TexturedLit { texture: GpuTextureHandle },
}
