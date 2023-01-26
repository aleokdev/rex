use glam::{vec3, Vec3};
use renderer::abs::mesh::{CpuMesh, Vertex};
use rex::grid::RoomId;

pub fn generate_test() -> CpuMesh {
    let mut mesh = CpuMesh::default();
    add_quad(&mut mesh, Vec3::ZERO, vec3(1., 0., 1.), Vec3::Y);
    mesh
}

pub fn generate_room_mesh(map: &rex::building::BuildingMap, id: RoomId) -> CpuMesh {
    let mut mesh = CpuMesh::default();
    for (&floor_idx, floor) in map.floors() {
        generate_room_mesh_for_floor(floor, floor_idx, id, &mut mesh)
    }

    mesh
}

pub fn generate_room_mesh_for_floor(
    floor: &rex::building::FloorMap,
    floor_idx: i32,
    id: RoomId,
    mesh: &mut CpuMesh,
) {
    generate_room_floor_mesh(floor, floor_idx, id, mesh);
    // TODO: Walls, doorways, ...
}

pub fn generate_room_floor_mesh(
    floor: &rex::building::FloorMap,
    floor_idx: i32,
    id: RoomId,
    mesh: &mut CpuMesh,
) {
    for position in floor.room_cell_positions()[id].iter() {
        let start = vec3(position.x as f32, floor_idx as f32, position.y as f32);
        add_quad(mesh, start, start + vec3(1., 0., 1.), Vec3::Y);
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
    mesh.vertices
        .extend(vs.into_iter().map(|position| Vertex { position, normal }));
    mesh.indices
        .extend([i0, i0 + 2, i0 + 3, i0 + 3, i0 + 2, i0 + 1]);
}
