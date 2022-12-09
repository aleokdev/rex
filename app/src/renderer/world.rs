use super::abs;

pub struct Camera {
    near: f32,
    far: f32,
    aspect: f32,
    fovy: f32,

    position: glam::Vec3,
    forward: glam::Vec3,
    right: glam::Vec3,
    up: glam::Vec3,

    yaw: f32,
    pitch: f32,
}

impl Camera {
    const LOOK_SENSITIVITY: f32 = 0.1;
    const MOVE_SPEED: f32 = 0.05;

    pub fn new(cx: &abs::Cx) -> Self {
        Camera {
            near: 0.1,
            far: 300.,
            aspect: cx.width as f32 / cx.height as f32,
            fovy: 60.0f32.to_radians(),

            position: glam::Vec3::ZERO,
            forward: glam::Vec3::new(1., 0., 0.),
            right: glam::Vec3::new(1., 0., 0.),
            up: glam::Vec3::new(0., 1., 0.),

            yaw: 0.,
            pitch: 0.,
        }
    }

    pub fn view(&self) -> glam::Mat4 {
        glam::Mat4::look_at_lh(self.position, self.position + self.forward, self.up)
    }

    pub fn proj(&self) -> glam::Mat4 {
        glam::Mat4::perspective_lh(self.fovy, self.aspect, self.near, self.far)
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        let dx = dx * Self::LOOK_SENSITIVITY;
        let dy = dy * Self::LOOK_SENSITIVITY;

        self.yaw += dx;
        self.pitch = (self.pitch + dy).clamp(-89., 89.);

        let dir = glam::Vec3::new(
            self.yaw.to_radians().cos() * self.pitch.to_radians().cos(),
            self.pitch.to_radians().sin(),
            self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
        );

        self.forward = dir.normalize();
    }

    pub fn move_forward(&mut self, x: f32) {
        self.position += self.forward * x * Self::MOVE_SPEED;
    }

    pub fn move_right(&mut self, x: f32) {
        self.position += self.forward.cross(self.up).normalize() * x * Self::MOVE_SPEED;
    }
}

pub struct World {
    pub camera: Camera,
    pub cube: glam::Vec3,
}
