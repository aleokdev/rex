use glam::{Mat4, Vec3, Vec4};

use super::abs;

pub struct Camera {
    near: f32,
    far: f32,
    aspect: f32,
    fovy: f32,

    position: glam::Vec3,
    forward: glam::Vec3,
    up: glam::Vec3,

    /// Camera rotation around the Y axis, in radians.
    /// A yaw of zero corresponds to looking in the +X axis direction.
    yaw: f32,
    /// Camera rotation around the X axis, in radians
    pitch: f32,
}

impl Camera {
    const PITCH_ANGLE_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.1;

    pub fn new(cx: &abs::Cx) -> Self {
        Camera {
            near: 0.1,
            far: 300.,
            aspect: cx.width as f32 / cx.height as f32,
            fovy: 60.0f32.to_radians(),

            position: glam::Vec3::ZERO,
            up: glam::Vec3::Y,
            forward: glam::Vec3::X,

            yaw: 0.,
            pitch: 0.,
        }
    }

    pub fn view(&self) -> glam::Mat4 {
        Mat4::from_cols(Vec4::X, Vec4::NEG_Y, Vec4::Z, Vec4::W)
            * glam::Mat4::look_at_lh(self.position, self.position + self.forward, self.up)
    }

    pub fn proj(&self) -> glam::Mat4 {
        glam::Mat4::perspective_lh(self.fovy, self.aspect, self.near, self.far)
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw += dx;
        self.pitch = (self.pitch + dy).clamp(-Self::PITCH_ANGLE_LIMIT, Self::PITCH_ANGLE_LIMIT);

        let dir = glam::Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        );

        self.forward = dir.normalize();
    }

    pub fn move_local_coords(&mut self, offset: Vec3) {
        self.position += self.forward * offset.z + self.up * offset.y + self.right() * offset.x;
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn right(&self) -> Vec3 {
        self.forward.cross(self.up).normalize()
    }

    pub fn forward(&self) -> Vec3 {
        self.forward
    }

    pub fn up(&self) -> Vec3 {
        self.up
    }
}

pub struct World {
    pub camera: Camera,
    pub cube: glam::Vec3,
}
