use ahash::AHashMap;
use glam::{vec2, IVec2, Vec2};

use crate::RoomId;

#[derive(Debug, Clone, Copy)]
pub struct SpaceAllocation {
    pos: Vec2,
    radius: f32,
    room_id: RoomId,
}

const CHUNK_SIZE: f32 = 4.;

#[derive(Debug, Clone, Default)]
pub struct Space {
    /// Each chunk contains the allocations that intersect with it.
    /// Each chunk is a `CHUNK_SIZE` sized square, starting from the origin.
    allocations: AHashMap<IVec2, Vec<SpaceAllocation>>,
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
        let chunk = self.mut_chunk_at(position);
        let is_colliding = chunk
            .iter()
            .any(|alloc| circle_collides(position, radius, alloc.pos, alloc.radius));

        if is_colliding {
            Err(AllocationError)
        } else {
            let x_range =
                (position.x - radius).floor() as i32..=(position.x + radius).ceil() as i32;
            let y_range =
                (position.y - radius).floor() as i32..=(position.y + radius).ceil() as i32;

            let allocation = SpaceAllocation {
                pos: position,
                radius,
                room_id: id,
            };

            for x in x_range {
                for y in y_range.clone() {
                    let pos = vec2(x as f32, y as f32);
                    if circle_intersects_rect(pos, vec2(CHUNK_SIZE, CHUNK_SIZE), position, radius) {
                        let chunk = self.mut_chunk_at(pos);
                        chunk.push(allocation);
                    }
                }
            }

            Ok(allocation)
        }
    }

    fn chunk_at(&self, position: Vec2) -> Option<&Vec<SpaceAllocation>> {
        self.allocations
            .get(&(position / CHUNK_SIZE).floor().as_ivec2())
    }

    fn mut_chunk_at(&mut self, position: Vec2) -> &mut Vec<SpaceAllocation> {
        self.allocations
            .entry((position / CHUNK_SIZE).floor().as_ivec2())
            .or_insert_with(|| vec![])
    }

    pub fn is_point_allocated(&self, position: Vec2) -> bool {
        self.chunk_at(position).map_or(false, |chunk| {
            chunk
                .iter()
                .any(|alloc| circle_contains(position, alloc.pos, alloc.radius))
        })
    }

    pub fn is_point_allocated_by_any_other_than(&self, id: RoomId, position: Vec2) -> bool {
        self.chunk_at(position).map_or(false, |chunk| {
            chunk
                .iter()
                .filter(|alloc| alloc.room_id != id)
                .any(|alloc| circle_contains(position, alloc.pos, alloc.radius))
        })
    }

    pub fn force_displacement(&self, position: Vec2) -> Vec2 {
        self.chunk_at(position).map_or(Vec2::ZERO, |chunk| {
            chunk.iter().fold(Vec2::ZERO, |acc, x| {
                let diff = position - x.pos;
                acc + (x.radius - diff.length()).max(0.) * diff
            })
        })
    }
}

fn circle_contains(pos: Vec2, circle_center: Vec2, r: f32) -> bool {
    (pos.x - circle_center.x) * (pos.x - circle_center.x)
        + (pos.y - circle_center.y) * (pos.y - circle_center.y)
        <= r * r
}

fn circle_collides(pos1: Vec2, r1: f32, pos2: Vec2, r2: f32) -> bool {
    (pos2 - pos1).length_squared() <= (r1 + r2) * (r1 + r2)
}

fn circle_intersects_rect(min: Vec2, size: Vec2, center: Vec2, r: f32) -> bool {
    let distance = (center - min).abs();

    if distance.x > (size.x / 2. + r) {
        return false;
    }
    if distance.y > (size.y / 2. + r) {
        return false;
    }

    if distance.x <= size.x / 2. {
        return true;
    }
    if distance.y <= size.y / 2. {
        return true;
    }

    let corner_distance_sq = (distance - size).length_squared();

    corner_distance_sq <= r * r
}
