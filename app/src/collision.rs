use glam::{ivec2, vec2, vec3, IVec2, IVec3, Vec2, Vec3};

use crate::meshgen::{CEILING_HEIGHT, LEVEL_HEIGHT, WALL_HALF_WIDTH};

pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Sphere {
    // https://stackoverflow.com/a/41457896
    pub fn intersects(&self, aabb: Aabb) -> bool {
        let mut dmin = 0.;
        for i in 0..3 {
            if self.center[i] < aabb.min[i] {
                dmin += (self.center[i] - aabb.min[i]) * (self.center[i] - aabb.min[i]);
            } else if self.center[i] > aabb.max[i] {
                dmin += (self.center[i] - aabb.max[i]) * (self.center[i] - aabb.max[i]);
            };
        }
        dmin <= self.radius * self.radius
    }
}

pub fn wall_collision_rect(dual_pos: IVec3) -> Aabb {
    let is_horizontal = (dual_pos.x + dual_pos.y) % 2 == 0;
    let corrected_dual_pos = if is_horizontal {
        dual_pos.truncate()
    } else {
        dual_pos.truncate() - IVec2::Y
    };
    let cell_pos = ivec2(
        (corrected_dual_pos.x - corrected_dual_pos.y) / 2,
        (corrected_dual_pos.x + corrected_dual_pos.y) / 2,
    );

    let (x, y) = (cell_pos.x, cell_pos.y);

    let mut from = if is_horizontal {
        vec2(x as f32, y as f32 - WALL_HALF_WIDTH)
    } else {
        vec2(x as f32 - WALL_HALF_WIDTH, y as f32)
    };

    let mut to = from
        + if is_horizontal {
            vec2(1., WALL_HALF_WIDTH * 2.)
        } else {
            vec2(WALL_HALF_WIDTH * 2., 1.)
        };

    Aabb {
        min: from.extend(dual_pos.z as f32 * LEVEL_HEIGHT),
        max: to.extend(dual_pos.z as f32 * LEVEL_HEIGHT + CEILING_HEIGHT),
    }
}
