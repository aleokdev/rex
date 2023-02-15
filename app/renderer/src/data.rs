use crate::{Camera, RenderObject};

pub struct RenderData {
    pub camera: Camera,
    pub objects: Vec<RenderObject>,
}
