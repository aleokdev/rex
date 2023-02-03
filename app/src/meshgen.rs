use common::coords;
use glam::{ivec2, vec3, IVec2, Vec3};
use renderer::abs::mesh::{CpuMesh, Vertex};
use rex::{
    building::{DualNormalDirection, Room},
    grid::RoomId,
    Database,
};

pub const DOOR_HEIGHT: f32 = 2.1;
pub const CEILING_HEIGHT: f32 = 3.4;
pub const LEVEL_HEIGHT: f32 = 3.5;

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
    generate_room_floor_ceiling_mesh(floor, floor_idx, id, mesh);
    // TODO: Doors
}

pub fn generate_room_floor_ceiling_mesh(
    floor: &rex::building::FloorMap,
    floor_idx: i32,
    id: RoomId,
    mesh: &mut CpuMesh,
) {
    if let Some(positions) = &floor.room_cell_positions().get(&id) {
        for position in positions.iter() {
            let start = vec3(
                position.x as f32,
                position.y as f32,
                floor_idx as f32 * LEVEL_HEIGHT,
            );
            add_quad(
                mesh,
                start,
                start + common::coords::SOUTH + common::coords::EAST,
                common::coords::UP,
                vec3(0.98, 0.3, 0.2),
            );
            add_quad(
                mesh,
                start + coords::UP * CEILING_HEIGHT,
                start + common::coords::SOUTH + common::coords::EAST + coords::UP * CEILING_HEIGHT,
                common::coords::DOWN,
                vec3(0.78, 0.9, 0.67),
            );
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct Wall;

pub fn generate_room_wall_mesh(room: &Room, mesh: &mut CpuMesh) {
    for (dual_pos, piece) in room.duals.iter() {
        const WALL_WIDTH: f32 = 0.1;
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
        match piece {
            rex::building::DualPiece::Wall { normal } => {
                let normal_vec;
                let from;

                if *normal == DualNormalDirection::NorthWest {
                    if is_horizontal {
                        from = vec3(x, y - WALL_WIDTH, dual_pos.z as f32 * LEVEL_HEIGHT);
                        normal_vec = common::coords::NORTH;
                    } else {
                        from = vec3(x - WALL_WIDTH, y, dual_pos.z as f32 * LEVEL_HEIGHT);
                        normal_vec = common::coords::WEST;
                    };
                } else {
                    if is_horizontal {
                        from = vec3(x, y + WALL_WIDTH, dual_pos.z as f32 * LEVEL_HEIGHT);
                        normal_vec = common::coords::SOUTH;
                    } else {
                        from = vec3(x + WALL_WIDTH, y, dual_pos.z as f32 * LEVEL_HEIGHT);
                        normal_vec = common::coords::EAST;
                    };
                }

                let to =
                    from + if is_horizontal {
                        common::coords::EAST
                    } else {
                        common::coords::SOUTH
                    } + common::coords::UP * CEILING_HEIGHT;

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
            rex::building::DualPiece::Door { .. } => {
                let from_se;
                let from_nw;
                let nw_normal_vector;
                let floor_height = dual_pos.z as f32 * LEVEL_HEIGHT;
                if is_horizontal {
                    from_se =
                        vec3(x, y + WALL_WIDTH, floor_height) + common::coords::UP * DOOR_HEIGHT;
                    from_nw =
                        vec3(x, y - WALL_WIDTH, floor_height) + common::coords::UP * DOOR_HEIGHT;
                    nw_normal_vector = common::coords::NORTH;
                } else {
                    from_se =
                        vec3(x + WALL_WIDTH, y, floor_height) + common::coords::UP * DOOR_HEIGHT;
                    from_nw =
                        vec3(x - WALL_WIDTH, y, floor_height) + common::coords::UP * DOOR_HEIGHT;
                    nw_normal_vector = common::coords::WEST;
                };

                let to_se = from_se
                    + if is_horizontal {
                        common::coords::EAST
                    } else {
                        common::coords::SOUTH
                    }
                    + common::coords::UP * (CEILING_HEIGHT - DOOR_HEIGHT);
                let to_nw = from_nw
                    + if is_horizontal {
                        common::coords::EAST
                    } else {
                        common::coords::SOUTH
                    }
                    + common::coords::UP * (CEILING_HEIGHT - DOOR_HEIGHT);

                // SE wall
                let vs = [
                    from_se,
                    to_se,
                    vec3(from_se.x, from_se.y, to_se.z),
                    vec3(to_se.x, to_se.y, from_se.z),
                ];
                let i0 = mesh.vertices.len() as u32;
                mesh.vertices.extend(vs.into_iter().map(|position| Vertex {
                    position,
                    normal: -nw_normal_vector,
                    color: Vec3::ONE,
                }));
                mesh.indices
                    .extend([i0, i0 + 2, i0 + 3, i0 + 3, i0 + 2, i0 + 1]);

                // NW wall
                let vs = [
                    from_nw,
                    to_nw,
                    vec3(from_nw.x, from_nw.y, to_nw.z),
                    vec3(to_nw.x, to_nw.y, from_nw.z),
                ];
                let i0 = mesh.vertices.len() as u32;
                mesh.vertices.extend(vs.into_iter().map(|position| Vertex {
                    position,
                    normal: nw_normal_vector,
                    color: Vec3::ONE,
                }));
                mesh.indices
                    .extend([i0, i0 + 2, i0 + 3, i0 + 3, i0 + 2, i0 + 1]);

                // Doorframe ceiling
                let vs = [
                    from_nw,
                    vec3(to_se.x, to_se.y, from_se.z),
                    vec3(from_nw.x, to_se.y, from_nw.z),
                    vec3(to_se.x, from_nw.y, from_nw.z),
                ];
                let i0 = mesh.vertices.len() as u32;
                mesh.vertices.extend(vs.into_iter().map(|position| Vertex {
                    position,
                    normal: coords::DOWN,
                    color: Vec3::ONE,
                }));
                mesh.indices
                    .extend([i0, i0 + 2, i0 + 3, i0 + 3, i0 + 2, i0 + 1]);

                // Doorframe sidings
                let vs = if is_horizontal {
                    [
                        vec3(from_nw.x, from_nw.y, floor_height),
                        vec3(from_nw.x, from_se.y, from_nw.z),
                        from_nw,
                        vec3(from_nw.x, from_se.y, floor_height),
                    ]
                } else {
                    [
                        vec3(from_nw.x, from_nw.y, floor_height),
                        vec3(from_se.x, from_nw.y, from_nw.z),
                        from_nw,
                        vec3(from_se.x, from_nw.y, floor_height),
                    ]
                };
                let i0 = mesh.vertices.len() as u32;
                mesh.vertices.extend(vs.into_iter().map(|position| Vertex {
                    position,
                    normal: if is_horizontal {
                        coords::EAST
                    } else {
                        coords::SOUTH
                    },
                    color: Vec3::ONE,
                }));
                mesh.indices
                    .extend([i0, i0 + 2, i0 + 3, i0 + 3, i0 + 2, i0 + 1]);

                let vs = if is_horizontal {
                    [
                        vec3(to_nw.x, to_nw.y, floor_height),
                        vec3(to_nw.x, to_se.y, to_nw.z),
                        to_nw,
                        vec3(to_nw.x, to_se.y, floor_height),
                    ]
                } else {
                    [
                        vec3(to_nw.x, to_nw.y, floor_height),
                        vec3(to_se.x, to_nw.y, to_nw.z),
                        to_nw,
                        vec3(to_se.x, to_nw.y, floor_height),
                    ]
                };
                let i0 = mesh.vertices.len() as u32;
                mesh.vertices.extend(vs.into_iter().map(|position| Vertex {
                    position,
                    normal: if is_horizontal {
                        coords::WEST
                    } else {
                        coords::NORTH
                    },
                    color: Vec3::ONE,
                }));
                mesh.indices
                    .extend([i0, i0 + 2, i0 + 3, i0 + 3, i0 + 2, i0 + 1]);
            }
            rex::building::DualPiece::Empty => continue,
        }
    }
}

pub fn add_quad(mesh: &mut CpuMesh, from: Vec3, to: Vec3, normal: Vec3, color: Vec3) {
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
        color,
    }));
    mesh.indices
        .extend([i0, i0 + 2, i0 + 3, i0 + 3, i0 + 2, i0 + 1]);
}
