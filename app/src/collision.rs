use glam::{ivec2, ivec3, vec2, vec3, IVec2, IVec3, Vec3};
use rex::Database;

use crate::meshgen::{CEILING_HEIGHT, LEVEL_HEIGHT, WALL_HALF_WIDTH};

pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
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

pub struct Quad {
    /// 0---1
    /// |   |
    /// |   |
    /// 3---2
    pub points: [Vec3; 4],
    pub normal: Vec3,
}

pub struct Aabb {
    /// Located at the west-north-down most position (-X-Y-Z)
    pub min: Vec3,
    /// Located at the east-south-up most position (+X+Y+Z)
    pub max: Vec3,
}

impl Aabb {
    pub fn east_quad(&self) -> Quad {
        Quad {
            points: [
                vec3(self.max.x, self.min.y, self.max.z),
                vec3(self.max.x, self.max.y, self.max.z),
                vec3(self.max.x, self.max.y, self.min.z),
                vec3(self.max.x, self.min.y, self.min.z),
            ],
            normal: common::coords::EAST,
        }
    }
    pub fn west_quad(&self) -> Quad {
        Quad {
            points: [
                vec3(self.min.x, self.min.y, self.max.z),
                vec3(self.min.x, self.max.y, self.max.z),
                vec3(self.min.x, self.max.y, self.min.z),
                vec3(self.min.x, self.min.y, self.min.z),
            ],
            normal: common::coords::WEST,
        }
    }
    pub fn north_quad(&self) -> Quad {
        Quad {
            points: [
                vec3(self.max.x, self.max.y, self.max.z),
                vec3(self.min.x, self.max.y, self.max.z),
                vec3(self.min.x, self.max.y, self.min.z),
                vec3(self.max.x, self.max.y, self.min.z),
            ],
            normal: common::coords::NORTH,
        }
    }
    pub fn south_quad(&self) -> Quad {
        Quad {
            points: [
                vec3(self.max.x, self.min.y, self.max.z),
                vec3(self.min.x, self.min.y, self.max.z),
                vec3(self.min.x, self.min.y, self.min.z),
                vec3(self.max.x, self.min.y, self.min.z),
            ],
            normal: common::coords::SOUTH,
        }
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

pub fn move_and_slide(database: &Database, starting_pos: Vec3, mut to_move: Vec3) -> Vec3 {
    fn calculate_delta_step(delta: Vec3) -> Vec3 {
        const MAX_STEP_LENGTH: f32 = 0.1;
        let delta_len = delta.length();
        if delta_len <= MAX_STEP_LENGTH {
            delta
        } else {
            delta / delta.length() * MAX_STEP_LENGTH
        }
    }
    let mut step = calculate_delta_step(to_move);

    let mut player_collider = Sphere {
        center: starting_pos,
        radius: 0.3,
    };

    while to_move.length() >= step.length() {
        let before_pos = player_collider.center;
        let target_pos = before_pos + step;
        to_move -= step;

        player_collider.center = target_pos;

        if collides_with_world(database, &player_collider) {
            // We've hit something, let's decompose our movement into X and Y axis to find what the
            // issue is

            player_collider.center.x = before_pos.x;
            if !collides_with_world(database, &player_collider) {
                // X was the problem

                to_move.x = 0.;
            } else {
                player_collider.center.x = target_pos.x;
                player_collider.center.y = before_pos.y;
                if !collides_with_world(database, &player_collider) {
                    // Y was the problem

                    to_move.y = 0.;
                } else {
                    // Both axes are the problem
                    player_collider.center = before_pos;
                    break;
                }
            }
            // TODO: Z collision checks as well
        }

        if to_move == Vec3::ZERO {
            break;
        } else {
            step = calculate_delta_step(to_move);
        }
    }

    player_collider.center
}

pub fn collides_with_world(database: &Database, collider: &Sphere) -> bool {
    let cell_pos = ivec3(
        collider.center.x.floor() as i32,
        collider.center.y.floor() as i32,
        (collider.center.z / CEILING_HEIGHT).floor() as i32,
    );

    if let Some(room_id) = database
        .map
        .floor(cell_pos.z)
        .and_then(|floor| floor.cell(cell_pos.truncate()))
    {
        let room = &database.rooms[room_id];
        let duals = &room.duals;
        for (&dual_pos, piece) in duals.iter() {
            match piece {
                rex::building::DualPiece::Wall { .. } => {
                    if collider.intersects(wall_collision_rect(dual_pos)) {
                        return true;
                    }
                }
                _ => (),
            }
        }

        false
    } else {
        // Everything outside the world counts as invisible block for now
        true
    }
}
