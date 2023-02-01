use glam::Vec3;

// These come directly from the Rex generator's coordinate system
// TODO: Export coordinate system in rex and reexport here
pub const EAST: Vec3 = Vec3::X;
pub const WEST: Vec3 = Vec3::NEG_X;
pub const SOUTH: Vec3 = Vec3::Y;
pub const NORTH: Vec3 = Vec3::NEG_Y;
pub const UP: Vec3 = Vec3::Z;
pub const DOWN: Vec3 = Vec3::NEG_Z;
