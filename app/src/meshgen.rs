use glam::{ivec2, vec3, IVec2, Vec3};
use renderer::abs::mesh::{CpuMesh, Vertex};
use rex::{
    building::{DualNormalDirection, Room},
    grid::{CartesianGrid, RoomId},
    Database,
};

pub fn generate_test() -> CpuMesh {
    let mut mesh = CpuMesh::default();
    add_quad(&mut mesh, Vec3::ZERO, vec3(1., 0., 1.), Vec3::Y);
    mesh
}

pub fn generate_room_mesh(db: &Database, id: RoomId) -> CpuMesh {
    let mut mesh = CpuMesh::default();
    for (&floor_idx, floor) in db.map.floors() {
        generate_room_mesh_for_floor(floor, floor_idx, id, &mut mesh)
    }
    generate_room_wall_mesh(&db.rooms[id], &mut mesh);

    mesh
}

pub fn generate_room_mesh_for_floor(
    floor: &rex::building::FloorMap,
    floor_idx: i32,
    id: RoomId,
    mesh: &mut CpuMesh,
) {
    generate_room_floor_mesh(floor, floor_idx, id, mesh);
    // TODO: Doors
}

pub fn generate_room_floor_mesh(
    floor: &rex::building::FloorMap,
    floor_idx: i32,
    id: RoomId,
    mesh: &mut CpuMesh,
) {
    if let Some(positions) = &floor.room_cell_positions().get(&id) {
        for position in positions.iter() {
            let start = vec3(position.x as f32, position.y as f32, floor_idx as f32);
            add_quad(
                mesh,
                start,
                start + common::coords::SOUTH + common::coords::EAST,
                common::coords::UP,
            );
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct Wall;

pub fn generate_room_wall_mesh(room: &Room, mesh: &mut CpuMesh) {
    for (dual_pos, piece) in room.duals.iter() {
        const WALL_WIDTH: f32 = 0.1;
        const WALL_HEIGHT: f32 = 2.8;
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
        let x = cell_pos.x as f32;
        let y = cell_pos.y as f32;
        let normal_vec;
        let from;
        match piece {
            rex::building::DualPiece::Wall { normal } => {
                if *normal == DualNormalDirection::NorthWest {
                    if is_horizontal {
                        from = vec3(x, y - WALL_WIDTH, dual_pos.z as f32 * WALL_HEIGHT);
                        normal_vec = common::coords::NORTH;
                    } else {
                        from = vec3(x - WALL_WIDTH, y, dual_pos.z as f32 * WALL_HEIGHT);
                        normal_vec = common::coords::WEST;
                    };
                } else {
                    if is_horizontal {
                        from = vec3(x, y + WALL_WIDTH, dual_pos.z as f32 * WALL_HEIGHT);
                        normal_vec = common::coords::SOUTH;
                    } else {
                        from = vec3(x + WALL_WIDTH, y, dual_pos.z as f32 * WALL_HEIGHT);
                        normal_vec = common::coords::EAST;
                    };
                }
            }
            rex::building::DualPiece::Door { .. } => continue,
            rex::building::DualPiece::Empty => continue,
        }
        let to =
            from + if is_horizontal {
                common::coords::EAST
            } else {
                common::coords::SOUTH
            } + common::coords::UP * WALL_HEIGHT;

        let vs = [
            from,
            to,
            vec3(from.x, from.y, to.z),
            vec3(to.x, to.y, from.z),
        ];

        let i0 = mesh.vertices.len() as u32;
        mesh.vertices.extend(vs.into_iter().map(|position| Vertex {
            position,
            normal: normal_vec,
            color: Vec3::ONE,
        }));
        mesh.indices
            .extend([i0, i0 + 2, i0 + 3, i0 + 3, i0 + 2, i0 + 1]);
    }
}

pub fn add_quad(mesh: &mut CpuMesh, from: Vec3, to: Vec3, normal: Vec3) {
    let diff = to - from;
    let diff_half = diff / 2.;
    let vs = [
        from,
        to,
        from + diff_half + normal.cross(diff_half),
        from + diff_half - normal.cross(diff_half),
    ];
    let i0 = mesh.vertices.len() as u32;
    mesh.vertices.extend(vs.into_iter().map(|position| Vertex {
        position,
        normal,
        color: Vec3::X,
    }));
    mesh.indices
        .extend([i0, i0 + 2, i0 + 3, i0 + 3, i0 + 2, i0 + 1]);
}
