use glam::{vec2, Vec2};

use crate::grid::RoomId;

#[derive(Debug, Clone, Copy)]
pub struct SpaceAllocation {
    pos: Vec2,
    radius: f32,
    room_id: RoomId,
}
#[derive(Debug, Clone, Default)]
pub struct Space {
    allocations: Vec<SpaceAllocation>,
}

/// The allocation didn't fit at the position given since part of the space it needed was already occupied.
#[derive(Debug, Clone)]
pub struct AllocationError;

impl Space {
    pub fn allocate_near(
        &mut self,
        id: RoomId,
        position: Vec2,
        starting_pos_radius: f32,
        radius: f32,
        rng: &mut impl rand::Rng,
    ) -> Vec2 {
        let random_angle = rng.gen_range(0.0..(std::f32::consts::PI * 2.));
        let random_dir = vec2(random_angle.cos(), random_angle.sin());
        const STEP: f32 = 1.;

        let mut try_pos = position + random_dir * starting_pos_radius;
        while let Err(_) = self.allocate_at(id, try_pos, radius) {
            try_pos += STEP * random_dir;
        }

        try_pos
    }

    pub fn allocate_at(
        &mut self,
        id: RoomId,
        position: Vec2,
        radius: f32,
    ) -> Result<SpaceAllocation, AllocationError> {
        let is_colliding = self
            .allocations
            .iter()
            .any(|alloc| circle_collides(position, radius, alloc.pos, alloc.radius));

        if is_colliding {
            Err(AllocationError)
        } else {
            let allocation = SpaceAllocation {
                pos: position,
                radius,
                room_id: id,
            };
            self.allocations.push(allocation);

            Ok(allocation)
        }
    }

    pub fn is_point_allocated(&self, position: Vec2) -> bool {
        self.allocations.iter().any(|alloc| {
            (position.x - alloc.pos.x) * (position.x - alloc.pos.x)
                + (position.y - alloc.pos.y) * (position.y - alloc.pos.y)
                <= alloc.radius
        })
    }

    pub fn is_point_allocated_by_any_other_than(&self, id: RoomId, position: Vec2) -> bool {
        self.allocations
            .iter()
            .filter(|alloc| alloc.room_id != id)
            .any(|alloc| {
                (position.x - alloc.pos.x) * (position.x - alloc.pos.x)
                    + (position.y - alloc.pos.y) * (position.y - alloc.pos.y)
                    <= alloc.radius
            })
    }
}

fn circle_collides(pos1: Vec2, r1: f32, pos2: Vec2, r2: f32) -> bool {
    (pos2 - pos1).length_squared() <= (r1 + r2) * (r1 + r2)
}
