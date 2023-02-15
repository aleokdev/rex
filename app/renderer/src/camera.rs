use glam::{Mat4, Vec3, Vec4};

use common::coords;

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
    /// Camera rotation around the X axis, in radians.
    pitch: f32,
}

impl Camera {
    const PITCH_ANGLE_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.1;
    /// A matrix that negates the Y axis (i.e. If up was previously -Y, by doing NEGATE_Y_MATRIX * PV we'll obtain up = +Y).
    const NEGATE_Y_MATRIX: Mat4 = Mat4::from_cols(Vec4::X, Vec4::NEG_Y, Vec4::Z, Vec4::W);

    pub fn new(position: Vec3, aspect_ratio: f32) -> Self {
        Camera {
            near: 0.1,
            far: 300.,
            aspect: aspect_ratio,
            fovy: 60.0f32.to_radians(),

            position,
            up: coords::UP,
            forward: coords::NORTH,

            yaw: 0.,
            pitch: 0.,
        }
    }

    pub fn view(&self) -> glam::Mat4 {
        Self::NEGATE_Y_MATRIX
            * glam::Mat4::look_at_lh(self.position, self.position + self.forward, self.up)
    }

    pub fn proj(&self) -> glam::Mat4 {
        glam::Mat4::perspective_lh(self.fovy, self.aspect, self.near, self.far)
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw += dx;
        self.pitch = (self.pitch + dy).clamp(-Self::PITCH_ANGLE_LIMIT, Self::PITCH_ANGLE_LIMIT);

        let dir = self.pitch.sin() * coords::UP
            + self.yaw.cos() * self.pitch.cos() * coords::NORTH
            + self.yaw.sin() * self.pitch.cos() * coords::NORTH.cross(coords::UP);

        self.forward = dir.normalize();
    }

    pub fn move_local_coords(&mut self, offset: Vec3) {
        self.position += self.forward.truncate().extend(0.).normalize() * offset.z
            + self.up * offset.y
            + self.right().truncate().extend(0.).normalize() * offset.x;
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn set_position(&mut self, position: glam::Vec3) {
        self.position = position;
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
