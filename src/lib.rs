pub mod building;
pub mod grid;
pub mod math;
pub mod node;
pub mod ser_map;
pub mod ser_set;
pub mod space;
pub mod wfc;

use building::{BuildingMap, DualNormalDirection, DualPiece, FloorIdx, FloorMap, Room};
pub use glam;
use node::{Node, NodeId, RexFile};
use serde::{Deserialize, Serialize};

use std::{collections::VecDeque, ops::ControlFlow};

use ahash::{AHashSet, HashSet};
use bitflags::bitflags;
use glam::{ivec2, uvec2, IVec2, IVec3, Vec2};
use grid::{CartesianGrid, CartesianRoomGrid, GridChunk, RoomId};
use rand::{seq::SliceRandom, Rng};
use space::SpaceAllocator;

// TODO: Custom result type
pub type Result<T> = anyhow::Result<T>;

/// Contains everything related to a rex universe - room positions, doors, teleporters, stairs, etc.
#[derive(Serialize, Deserialize)]
pub struct Database {
    pub map: BuildingMap,
    pub nodes: Vec<Node>,
    pub rooms: Vec<Room>,
}

pub struct V3 {
    /// Indicates the nodes that the algorithm has yet to process.
    /// They will be processed from front to back, adding new items to the back.
    queue: VecDeque<usize>,
    database: Database,
    allocator: SpaceAllocator,
    teleports_used: usize,
}

pub struct V3Data {
    pub database: Database,
    pub teleports_used: usize,
}

impl V3 {
    pub fn new(mut nodes: Vec<Node>) -> Self {
        let mut queue = VecDeque::new();
        queue.push_back(0);
        let mut map = BuildingMap::default();
        let floor = map.floor_entry(0).or_default();
        // We will allocate at least one room per node, so we preallocate for performance's sake
        let mut rooms = Vec::with_capacity(nodes.len());
        let root_room = Self::allocate_room(&mut rooms, &mut nodes[0].rooms, 0, IVec3::ZERO);
        floor.assign_cell(IVec2::ZERO, root_room);
        expand_rooms(floor, [(0, IVec2::ZERO)].into_iter());
        Self {
            // We start on the root node
            queue,
            allocator: SpaceAllocator::default(),
            database: Database { map, rooms, nodes },
            teleports_used: 0,
        }
    }

    pub fn iterate(mut self, rng: &mut impl Rng) -> ControlFlow<V3Data, Self> {
        // Take the next node to process, if any. Otherwise we break since we've finished.
        let Some(node_idx) = self.queue.pop_front() else { return ControlFlow::Break(V3Data { database: self.database, teleports_used: self.teleports_used } ) };
        let Database {
            nodes,
            map: floors,
            rooms,
        } = &mut self.database;
        // Fetch some required node data.
        let mut room_id_being_expanded = *nodes[node_idx].rooms.first().unwrap();
        let room_pos = rooms[room_id_being_expanded].starting_pos();
        let mut current_floor = room_pos.z;
        let mut room_pos = room_pos.truncate();

        // Fetch the floor the room should be in, or instantiate it if it doesn't exist yet.
        let floor = floors.floor_entry(current_floor).or_default();
        let mut edge_positions: Vec<AdjacentCell> =
            AdjacentCellsIter::new(floor, room_id_being_expanded, room_pos)
                .filter(|&AdjacentCell { to, .. }| floor.cell(to).is_none())
                .collect();
        let mut positions_to_expand_to = Vec::new();
        let mut child_idx = 0;
        let mut first_child_idx_to_expand = 0;
        let mut num_of_children_to_expand = 0;
        let mut rooms_touched = vec![room_id_being_expanded];
        while child_idx < nodes[node_idx].children.len() {
            while edge_positions.len() == 0 {
                // Try to expand this room.
                if let Some(AdjacentCell { to: pos, .. }) =
                    positions_to_expand_to.choose(rng).copied()
                {
                    // If we have space in this floor, expand this room at the same time as the
                    // children instantiated.
                    let position_to_expand_to = pos;
                    expand_rooms(
                        floors.floor_entry(current_floor).or_default(),
                        (first_child_idx_to_expand
                            ..first_child_idx_to_expand + num_of_children_to_expand)
                            .map(|child_idx| nodes[node_idx].children[child_idx])
                            .map(|child_node_id| {
                                self.queue.push_back(child_node_id);
                                let room_id = *nodes[child_node_id].rooms.first().unwrap();
                                (room_id, rooms[room_id].starting_pos().truncate())
                            })
                            .chain(std::iter::once((
                                room_id_being_expanded,
                                position_to_expand_to,
                            ))),
                    );
                    first_child_idx_to_expand =
                        first_child_idx_to_expand + num_of_children_to_expand;
                    num_of_children_to_expand = 0;
                } else {
                    // If we don't have any room to expand to, we try to expand to the upper or
                    // lower floor.
                    // First expand the children instantiated in this floor.
                    expand_rooms(
                        floors.floor_entry(current_floor).or_default(),
                        (first_child_idx_to_expand
                            ..first_child_idx_to_expand + num_of_children_to_expand)
                            .map(|child_idx| nodes[node_idx].children[child_idx])
                            .map(|child_node_id| {
                                self.queue.push_back(child_node_id);
                                let room_id = *nodes[child_node_id].rooms.first().unwrap();
                                (room_id, rooms[room_id].starting_pos().truncate())
                            }),
                    );
                    // And generate walls, before we move onto a new room.
                    Self::calculate_walls(
                        &mut rooms[room_id_being_expanded],
                        room_id_being_expanded,
                        floors.floor_entry(current_floor).or_default(),
                        current_floor,
                    );
                    first_child_idx_to_expand =
                        first_child_idx_to_expand + num_of_children_to_expand;
                    num_of_children_to_expand = 0;
                    let upper_floor = floors.floor_entry(current_floor + 1).or_default();
                    // TODO: Take random position within the confines of the room instead of its starting pos for expanding up/down
                    if upper_floor.cell(room_pos).is_none() {
                        current_floor += 1;
                    } else {
                        let lower_floor = floors.floor_entry(current_floor - 1).or_default();
                        if lower_floor.cell(room_pos).is_none() {
                            current_floor -= 1;
                        } else {
                            // No room: We insert a teleport link. Move by 10 chunks at a time until
                            // we find an unused chunk, then expand there.
                            let mut current_chunk =
                                CartesianRoomGrid::global_coords_to_chunk_and_local_coords(
                                    room_pos,
                                )
                                .0;
                            let dir = [IVec2::X, IVec2::Y, IVec2::NEG_X, IVec2::NEG_Y]
                                [rng.gen_range(0..4)];

                            current_chunk += dir * 10;
                            let floor = floors.floor_mut(current_floor).unwrap();
                            while floor.grid().chunks.contains_key(&current_chunk) {
                                current_chunk += dir * 10;
                            }

                            let random_chunk_pos = uvec2(
                                rng.gen_range(0..GridChunk::<Option<usize>>::CELLS_PER_AXIS),
                                rng.gen_range(0..GridChunk::<Option<usize>>::CELLS_PER_AXIS),
                            );

                            room_pos = IVec2::from(
                                CartesianRoomGrid::chunk_and_local_coords_to_global_coords(
                                    current_chunk,
                                    random_chunk_pos,
                                ),
                            );
                            self.teleports_used += 1;
                        }
                    }

                    // Create a new room
                    room_id_being_expanded = Self::allocate_room(
                        rooms,
                        &mut nodes[node_idx].rooms,
                        node_idx,
                        room_pos.extend(current_floor),
                    );
                    expand_rooms(
                        floors.floor_entry(current_floor).or_default(),
                        std::iter::once((room_id_being_expanded, room_pos)),
                    );
                    rooms_touched.push(room_id_being_expanded);
                };

                let floor = floors.floor_entry(current_floor).or_default();
                positions_to_expand_to.clear();
                edge_positions = AdjacentCellsIter::new(floor, room_id_being_expanded, room_pos)
                    .filter(|&AdjacentCell { to, .. }| floor.cell(to).is_none())
                    .collect();
            }

            // Allocate a room for the next child
            let AdjacentCell {
                from: pos_allocated_from,
                to: pos_allocated_to,
            } = edge_positions.pop().unwrap();
            let child_starting_pos = pos_allocated_to.extend(current_floor);
            let child_node_id = nodes[node_idx].children[child_idx];
            let child_room = Self::allocate_room(
                rooms,
                &mut nodes[child_node_id].rooms,
                child_node_id,
                child_starting_pos,
            );
            num_of_children_to_expand += 1;
            child_idx += 1;
            // Door placement
            let door_dir = Direction::try_from(pos_allocated_to - pos_allocated_from).unwrap();
            let door_normal = match door_dir {
                Direction::North | Direction::West => DualNormalDirection::SouthEast,
                Direction::South | Direction::East => DualNormalDirection::NorthWest,
            };
            let door_pos = cell_to_wall_space(pos_allocated_from, door_dir).extend(current_floor);
            // On parent
            rooms[room_id_being_expanded].duals.insert(
                door_pos,
                DualPiece::Door {
                    normal: door_normal,
                },
            );
            rooms[room_id_being_expanded]
                .connections
                .push(pos_allocated_from.extend(current_floor));
            // On child
            rooms[child_room].duals.insert(door_pos, DualPiece::Empty);
            rooms[child_room]
                .connections
                .push(pos_allocated_to.extend(current_floor));
            if let Some(x) = edge_positions.pop() {
                positions_to_expand_to.push(x);
            }
            if let Some(x) = edge_positions.pop() {
                positions_to_expand_to.push(x);
            }
            if let Some(x) = edge_positions.pop() {
                positions_to_expand_to.push(x);
            }
            if let Some(x) = edge_positions.pop() {
                positions_to_expand_to.push(x);
            }
        }

        expand_rooms(
            floors.floor_entry(current_floor).or_default(),
            (first_child_idx_to_expand..first_child_idx_to_expand + num_of_children_to_expand)
                .map(|child_idx| nodes[node_idx].children[child_idx])
                .map(|child_node_id| {
                    self.queue.push_back(child_node_id);
                    let room_id = *nodes[child_node_id].rooms.first().unwrap();
                    (room_id, rooms[room_id].starting_pos().truncate())
                }),
        );

        // Once we have expanded all children, we can continue by placing all files that should be in the rooms we have
        // created.
        // First, we calculate which positions are available to place files in.
        // To do this, we take all the cells we have expanded to, and remove cells that are part of paths from one door
        // to another.
        for room_id in rooms_touched {
            let room = &mut rooms[room_id];
            for (&floor_idx, _floor, cell_positions) in
                floors.floors().filter_map(|(floor_idx, floor)| {
                    floor
                        .room_cell_positions()
                        .get(&room_id)
                        .map(|x| (floor_idx, floor, x))
                })
            {
                let mut available_cells: AHashSet<IVec2> = cell_positions.clone();
                for end in room.connections.iter() {
                    if let Some((path, _)) = pathfinding::directed::astar::astar(
                        &room.starting_pos().truncate(),
                        |&pos| {
                            [
                                pos - IVec2::X,
                                pos + IVec2::X,
                                pos - IVec2::Y,
                                pos + IVec2::Y,
                            ]
                            .into_iter()
                            .map(|pos| (pos, if cell_positions.contains(&pos) { 1 } else { 50 }))
                        },
                        |pos| (pos.x - end.x) * (pos.x - end.x) + (pos.y - end.y) * (pos.y - end.y),
                        |&pos| pos == end.truncate(),
                    ) {
                        for cell in path {
                            available_cells.remove(&cell);
                        }
                    }
                }
                room.available_cells
                    .extend(available_cells.into_iter().map(|x| x.extend(floor_idx)));
            }

            // TODO: Expand if cannot fit all files into allocated space
            let positions_to_use: Vec<IVec3> = room
                .available_cells
                .iter()
                .take(nodes[node_idx].files.len())
                .copied()
                .collect();
            for (file, pos) in nodes[node_idx].files.iter().zip(positions_to_use.iter()) {
                room.available_cells.remove(&pos);
                room.cells.insert(*pos, file.clone());
            }
        }

        Self::calculate_walls(
            &mut rooms[room_id_being_expanded],
            room_id_being_expanded,
            floors.floor_entry(current_floor).or_default(),
            current_floor,
        );

        ControlFlow::Continue(self)
    }

    pub fn allocator(&self) -> &SpaceAllocator {
        &self.allocator
    }

    pub fn teleports_used(&self) -> usize {
        self.teleports_used
    }

    pub fn database(&self) -> &Database {
        &self.database
    }

    fn allocate_room(
        rooms: &mut Vec<Room>,
        node_rooms: &mut Vec<RoomId>,
        node_id: NodeId,
        starting_pos: IVec3,
    ) -> RoomId {
        let mut room = Room::new(node_id, starting_pos);
        rooms.push(room);
        let id = rooms.len() - 1;
        node_rooms.push(id);
        id
    }

    // XXX: This only works with single floor rooms
    fn calculate_walls(room: &mut Room, room_id: RoomId, map: &FloorMap, floor_idx: FloorIdx) {
        let Some(owned_cells) = map.room_cell_positions().get(&room_id) else { return; };
        for &cell_pos in owned_cells.iter() {
            let directions = [
                (Direction::North, DualNormalDirection::SouthEast),
                (Direction::West, DualNormalDirection::SouthEast),
                (Direction::South, DualNormalDirection::NorthWest),
                (Direction::East, DualNormalDirection::NorthWest),
            ];
            directions
                .into_iter()
                .filter(|(offset, _)| map.cell(cell_pos + IVec2::from(*offset)) != Some(room_id))
                .for_each(|(dir, normal)| {
                    let pos = cell_to_wall_space(cell_pos, dir).extend(floor_idx);
                    let corners = [
                        map.cell(cell_pos + IVec2::from(dir) + IVec2::from(dir.ccw_rotated()))
                            == Some(room_id),
                        map.cell(cell_pos + IVec2::from(dir) + IVec2::from(dir.cw_rotated()))
                            == Some(room_id),
                    ];
                    room.duals
                        .entry(pos)
                        .or_insert(building::DualPiece::Wall { normal, corners });
                })
        }
    }
}

#[derive(Clone, Copy)]
struct AdjacentCell {
    from: IVec2,
    to: IVec2,
}

struct AdjacentCellsIter<'s> {
    map: &'s FloorMap,
    id: RoomId,
    starting_pos: IVec2,
    edge_pos: IVec2,
    direction: IVec2,
    done: bool,
}

impl<'s> AdjacentCellsIter<'s> {
    fn new(map: &'s FloorMap, id: RoomId, room_pos: IVec2) -> Self {
        // Find the room edge by moving towards +X until we aren't in the room any more
        let mut room_edge = room_pos;
        while map.cell(room_edge) == Some(id) {
            room_edge.x += 1;
        }

        Self {
            map,
            id,
            starting_pos: room_edge,
            edge_pos: room_edge,
            direction: IVec2::Y,
            done: false,
        }
    }
}

impl Iterator for AdjacentCellsIter<'_> {
    type Item = AdjacentCell;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        assert!(self.map.cell(self.edge_pos) != Some(self.id));

        fn rotate_cw(v: IVec2) -> IVec2 {
            ivec2(-v.y, v.x)
        }
        fn rotate_ccw(v: IVec2) -> IVec2 {
            ivec2(v.y, -v.x)
        }
        // If about to hit the room, rotate counterclockwise (and go backwards if appropiate)
        if self.map.cell(self.edge_pos + self.direction) == Some(self.id) {
            while self.map.cell(self.edge_pos + rotate_ccw(self.direction)) == Some(self.id) {
                // #|-
                // #^#
                // ###
                self.edge_pos -= self.direction;
            }
            // else
            // #|
            // #L_
            // ###
            self.direction = rotate_ccw(self.direction);
        }

        let result = self.edge_pos;
        let from = result + rotate_cw(self.direction);
        self.edge_pos += self.direction;

        // Skip outer corners
        if self.map.cell(self.edge_pos + rotate_cw(self.direction)) != Some(self.id) {
            // #|
            // #|
            // -+
            self.direction = rotate_cw(self.direction);
            self.edge_pos += self.direction;
        }

        if self.edge_pos == self.starting_pos {
            self.done = true;
        };

        Some(AdjacentCell { from, to: result })
    }
}

struct RoomExpansionPositions {
    nw_corner: IVec2,
    se_corner: IVec2,
    expand_left: bool,
    expand_right: bool,
    expand_up: bool,
    expand_down: bool,
    longest_side_edge: usize,
}

/// Expands a set of rooms given by an iterator from the position given.
///
/// They will expand until they collide with other rooms or they surpass an expansion limit.
fn expand_rooms(floor: &mut FloorMap, rooms: impl Iterator<Item = (usize, IVec2)>) {
    const MAX_LONGEST_SIDE_EDGE: usize = 7;

    let mut expandable: Vec<(usize, RoomExpansionPositions)> = rooms
        .map(|(id, pos)| {
            floor.assign_cell(pos, id);
            (
                id,
                RoomExpansionPositions {
                    nw_corner: pos,
                    se_corner: pos,
                    expand_down: true,
                    expand_left: true,
                    expand_right: true,
                    expand_up: true,
                    longest_side_edge: 1,
                },
            )
        })
        .collect();

    while !expandable.is_empty() {
        expandable.retain_mut(|&mut (id, ref mut positions)| {
            if positions.expand_left {
                let positions_to_expand_to = (positions.nw_corner.y..=positions.se_corner.y)
                    .map(|y| ivec2(positions.nw_corner.x - 1, y));
                if positions_to_expand_to
                    .clone()
                    .all(|pos| floor.cell(pos).map(|cell| cell == id).unwrap_or(true))
                {
                    positions_to_expand_to.for_each(|pos| floor.assign_cell(pos, id));
                    positions.nw_corner.x -= 1;
                } else {
                    positions.expand_left = false;
                }
            }
            if positions.expand_right {
                let positions_to_expand_to = (positions.nw_corner.y..=positions.se_corner.y)
                    .map(|y| ivec2(positions.se_corner.x + 1, y));
                if positions_to_expand_to
                    .clone()
                    .all(|pos| floor.cell(pos).map(|cell| cell == id).unwrap_or(true))
                {
                    positions_to_expand_to.for_each(|pos| floor.assign_cell(pos, id));
                    positions.se_corner.x += 1;
                } else {
                    positions.expand_right = false;
                }
            }
            if positions.expand_up {
                let positions_to_expand_to = (positions.nw_corner.x..=positions.se_corner.x)
                    .map(|x| ivec2(x, positions.nw_corner.y - 1));
                if positions_to_expand_to
                    .clone()
                    .all(|pos| floor.cell(pos).map(|cell| cell == id).unwrap_or(true))
                {
                    positions_to_expand_to.for_each(|pos| floor.assign_cell(pos, id));
                    positions.nw_corner.y -= 1;
                } else {
                    positions.expand_up = false;
                }
            }
            if positions.expand_down {
                let positions_to_expand_to = (positions.nw_corner.x..=positions.se_corner.x)
                    .map(|x| ivec2(x, positions.se_corner.y + 1));
                if positions_to_expand_to
                    .clone()
                    .all(|pos| floor.cell(pos).map(|cell| cell == id).unwrap_or(true))
                {
                    positions_to_expand_to.for_each(|pos| floor.assign_cell(pos, id));
                    positions.se_corner.y += 1;
                } else {
                    positions.expand_down = false;
                }
            }

            positions.longest_side_edge += 1;

            // Retain any nodes that can still expand in any direction
            positions.longest_side_edge < MAX_LONGEST_SIDE_EDGE
                && (positions.expand_down
                    || positions.expand_left
                    || positions.expand_right
                    || positions.expand_up)
        });
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    // -Y
    North,
    // +X
    East,
    // +Y
    South,
    // -X
    West,
}

impl Direction {
    #[allow(dead_code)]
    pub fn opposite(self) -> Self {
        match self {
            Direction::North => Direction::South,
            Direction::East => Direction::West,
            Direction::South => Direction::North,
            Direction::West => Direction::East,
        }
    }

    pub fn cw_rotated(self) -> Direction {
        match self {
            Direction::North => Direction::East,
            Direction::East => Direction::South,
            Direction::South => Direction::West,
            Direction::West => Direction::North,
        }
    }
    pub fn ccw_rotated(self) -> Direction {
        match self {
            Direction::North => Direction::West,
            Direction::East => Direction::North,
            Direction::South => Direction::East,
            Direction::West => Direction::South,
        }
    }
}

impl TryFrom<IVec2> for Direction {
    type Error = ();

    fn try_from(value: IVec2) -> std::result::Result<Self, Self::Error> {
        match value {
            IVec2::X => Ok(Direction::East),
            IVec2::Y => Ok(Direction::South),
            IVec2::NEG_Y => Ok(Direction::North),
            IVec2::NEG_X => Ok(Direction::West),
            _ => Err(()),
        }
    }
}

impl From<Direction> for IVec2 {
    fn from(value: Direction) -> Self {
        match value {
            Direction::North => IVec2::NEG_Y,
            Direction::East => IVec2::X,
            Direction::South => IVec2::Y,
            Direction::West => IVec2::NEG_X,
        }
    }
}

pub fn cell_to_wall_space(pos: IVec2, direction: Direction) -> IVec2 {
    //   x   x
    // x o x o x
    //   x   x
    // x o x o x
    //   x   x
    // o: cells, x: walls
    // wall origin is north wall of the (0,0) cell
    // the axes are rotated 45º with respect to the cell grid so that the Y axis points to the
    // southwest and the X axis points to the southeast
    ivec2(pos.y + pos.x, pos.y - pos.x)
        + match direction {
            Direction::North => IVec2::ZERO,
            Direction::East => IVec2::X,
            Direction::South => IVec2::ONE,
            Direction::West => IVec2::Y,
        }
}

#[test]
fn test_wall_space() {
    assert_eq!(
        cell_to_wall_space(ivec2(1, 0), Direction::West),
        ivec2(1, 0)
    );
    assert_eq!(
        cell_to_wall_space(ivec2(1, 1), Direction::North),
        ivec2(2, 0)
    );
    assert_eq!(
        cell_to_wall_space(ivec2(-1, 0), Direction::East),
        ivec2(0, 1)
    );
    assert_eq!(
        cell_to_wall_space(ivec2(1, 2), Direction::South),
        ivec2(4, 2)
    );
}

pub fn door_to_wall_space(pos: IVec2, direction: Direction) -> IVec2 {
    let t = pos
        - match direction {
            Direction::North => IVec2::ZERO,
            Direction::East => IVec2::X,
            Direction::South => IVec2::ONE,
            Direction::West => IVec2::Y,
        };
    ivec2((t.x - t.y) / 2, (t.x + t.y) / 2)
}

pub struct Door {
    /// Position of the door pivot.
    pub position: Vec2,
    /// Rotation of the door in radians. With rotation = 0, the door should face towards +Y.
    pub rotation: f32,
}

pub fn generate_doors(
    room_positions: &[IVec2],
    room_map: &CartesianRoomGrid,
    nodes: &[Node],
) -> Vec<Door> {
    let mut doors = Vec::with_capacity(nodes.len());
    let mut to_connect: Vec<HashSet<usize>> = nodes
        .iter()
        .map(|node| node.children.iter().copied().collect())
        .collect();
    for (id, &room_pos) in room_positions.iter().enumerate() {
        // Find the room edge by moving towards +X until we aren't in the room any more
        let mut room_edge = room_pos;
        while room_map.cell(room_edge) == Some(&Some(id)) {
            room_edge.x += 1;
        }

        let starting_pos = room_edge;

        // Move clockwise, connecting where possible, until we either connect to all children or we
        // reach the starting position again
        let mut direction = IVec2::Y;
        loop
        /*&& !to_connect[id].is_empty()*/
        {
            assert!(room_map.cell(room_edge) != Some(&Some(id)));

            let edge_cell = room_map.cell(room_edge).copied().flatten();
            if let Some(edge_cell) = edge_cell {
                if to_connect[id].contains(&edge_cell) {
                    to_connect[id].remove(&edge_cell);
                    doors.push(Door {
                        position: room_edge.as_vec2(),
                        rotation: (direction.y as f32).atan2(direction.x as f32),
                    })
                }
            }
            fn rotate_cw(v: IVec2) -> IVec2 {
                ivec2(-v.y, v.x)
            }
            fn rotate_ccw(v: IVec2) -> IVec2 {
                ivec2(v.y, -v.x)
            }
            // If about to hit the room, rotate counterclockwise
            if room_map.cell(room_edge + direction) == Some(&Some(id)) {
                if room_map.cell(room_edge + rotate_ccw(direction)) == Some(&Some(id)) {
                    // #|-
                    // #^#
                    // ###
                    room_edge -= direction;
                }
                // else
                // #|
                // #L_
                // ###
                direction = rotate_ccw(direction);
            } else if room_map.cell(room_edge + rotate_cw(direction)) != Some(&Some(id)) {
                // #|
                // #|
                // -+
                direction = rotate_cw(direction);
            }
            room_edge += direction;

            if room_edge == starting_pos {
                break;
            }
        }
    }

    doors
}
