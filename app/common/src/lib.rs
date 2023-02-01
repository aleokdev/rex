mod camera;
pub mod coords;
pub use camera::Camera;

pub struct World {
    pub camera: Camera,
}
