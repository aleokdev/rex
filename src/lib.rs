pub mod grid;
pub mod space;

pub use glam;

use std::{
    collections::VecDeque,
    fs,
    ops::{ControlFlow, Range},
    path::{Path, PathBuf},
};

use ahash::{AHashMap, HashSet};
use bitflags::bitflags;
use glam::{ivec2, vec2, IVec2, IVec3, Vec2};
use grid::{CartesianGrid, CartesianRoomGrid, RoomTemplate};
use rand::{seq::SliceRandom, Rng};
use space::{SpaceAllocation, SpaceAllocator};

// TODO: Custom result type
pub type Result<T> = anyhow::Result<T>;

#[derive(Default)]
pub struct Room {
    pub path: PathBuf,
    /// Counter-clockwise points
    pub mesh: Vec<Vec2>,
    pub connections: Vec<Connection>,
}

pub struct Connection {
    /// Global grid pos
    start: IVec2,
    /// Global grid pos
    end: IVec2,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub path: PathBuf,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
}

pub fn generate_nodes(path: &Path) -> Result<Vec<Node>> {
    let mut nodes = Vec::<Node>::with_capacity(100);

    let mut to_process = vec![Node {
        path: path.to_owned(),
        parent: None,
        children: vec![],
    }];

    while let Some(node_being_processed) = to_process.pop() {
        let node_being_processed_idx = nodes.len();

        if let Some(parent) = node_being_processed.parent {
            nodes[parent].children.push(node_being_processed_idx);
        }

        for dir_entry in fs::read_dir(&node_being_processed.path)? {
            let dir_entry = dir_entry?;

            if dir_entry.file_type()?.is_dir() {
                to_process.push(Node {
                    path: dir_entry.path(),
                    parent: Some(node_being_processed_idx),
                    children: vec![],
                });
            }
        }
        nodes.push(node_being_processed);
    }

    Ok(nodes)
}

#[derive(Default)]
pub struct RoomTemplateDb {
    templates: Vec<RoomTemplate>,
}

impl RoomTemplateDb {
    pub fn random(&self, rng: &mut impl Rng) -> Option<&RoomTemplate> {
        self.templates.choose(rng)
    }

    pub fn push(&mut self, template: RoomTemplate) {
        self.templates.push(template)
    }
}

pub struct V3 {
    nodes: Vec<Node>,
    room_positions: Vec<Option<IVec3>>,
    /// Indicates the node branch the algorithm has gone through.
    /// It will unwind when there are no more children to process in a given node,
    /// and increase its size when a node has children to process.
    queue: VecDeque<usize>,
    floors: AHashMap<i32, CartesianRoomGrid>,
    allocator: SpaceAllocator,
}

impl V3 {
    pub fn new(nodes: Vec<Node>) -> Self {
        let mut queue = VecDeque::new();
        queue.push_back(0);
        let mut floors = AHashMap::from_iter([(0, CartesianRoomGrid::default())]);
        floors.entry(0).or_default().set_cell(IVec2::ZERO, Some(0));
        expand_room(floors.get_mut(&0).unwrap(), [(0, IVec2::ZERO)].into_iter());
        let mut room_positions = vec![None; nodes.len()];
        room_positions[0] = Some(IVec3::ZERO);
        Self {
            room_positions,
            nodes,
            // We start on the root node
            queue,
            allocator: SpaceAllocator::default(),
            floors,
        }
    }

    pub fn iterate(&mut self, rng: &mut impl Rng) -> ControlFlow<(), ()> {
        let Some(node_idx) = self.queue.pop_front() else { return ControlFlow::Break(()) };
        let node = &self.nodes[node_idx];
        let room_pos = self.room_positions[node_idx].unwrap();
        let mut current_floor = room_pos.z;
        let room_pos = room_pos.truncate();

        let floor = self.floors.entry(current_floor).or_default();
        let mut edge_positions: Vec<IVec2> = AdjacentCellsIter::new(floor, node_idx, room_pos)
            .filter(|&pos| floor.cell(pos).copied().flatten().is_none())
            .collect();
        let mut positions_to_expand_to = Vec::new();
        let mut child_idx = 0;
        let mut to_expand = Vec::new();
        while edge_positions.len() > 0 && child_idx < node.children.len() {
            self.room_positions[node.children[child_idx]] =
                Some(edge_positions.pop().unwrap().extend(current_floor));
            to_expand.push(child_idx);
            child_idx += 1;
            if let Some(x) = edge_positions.pop() {
                positions_to_expand_to.push(x);
            }
            if let Some(x) = edge_positions.pop() {
                positions_to_expand_to.push(x);
            }
            if edge_positions.len() == 0 {
                if let Some(pos) = positions_to_expand_to.choose(rng).copied() {
                    let position_to_expand_to = pos;
                    expand_room(
                        self.floors.entry(current_floor).or_default(),
                        to_expand
                            .drain(..)
                            .map(|child_idx| node.children[child_idx])
                            .map(|child| {
                                self.queue.push_back(child);
                                (child, self.room_positions[child].unwrap().truncate())
                            })
                            .chain(std::iter::once((node_idx, position_to_expand_to))),
                    );
                } else {
                    expand_room(
                        self.floors.entry(current_floor).or_default(),
                        to_expand
                            .drain(..)
                            .map(|child_idx| node.children[child_idx])
                            .map(|child| {
                                self.queue.push_back(child);
                                (child, self.room_positions[child].unwrap().truncate())
                            }),
                    );
                    let upper_floor = self.floors.entry(current_floor + 1).or_default();
                    if upper_floor.cell(room_pos).is_none() {
                        current_floor += 1;
                    } else {
                        let lower_floor = self.floors.entry(current_floor - 1).or_default();
                        if lower_floor.cell(room_pos).is_none() {
                            current_floor -= 1;
                        } else {
                            break;
                        }
                    }
                    expand_room(
                        self.floors.entry(current_floor).or_default(),
                        std::iter::once((node_idx, room_pos)),
                    );
                };

                let floor = self.floors.entry(current_floor).or_default();
                positions_to_expand_to.clear();
                edge_positions = AdjacentCellsIter::new(floor, node_idx, room_pos)
                    .filter(|&pos| floor.cell(pos).copied().flatten().is_none())
                    .collect();
            }
        }

        expand_room(
            self.floors.entry(current_floor).or_default(),
            to_expand
                .iter()
                .map(|&child_idx| node.children[child_idx])
                .map(|child| {
                    self.queue.push_back(child);
                    (child, self.room_positions[child].unwrap().truncate())
                }),
        );

        ControlFlow::Continue(())
    }

    pub fn nodes(&self) -> &[Node] {
        self.nodes.as_ref()
    }

    pub fn room_positions(&self) -> &[Option<IVec3>] {
        self.room_positions.as_ref()
    }

    pub fn allocator(&self) -> &SpaceAllocator {
        &self.allocator
    }

    pub fn map(&self) -> &AHashMap<i32, CartesianRoomGrid> {
        &self.floors
    }
}

struct AdjacentCellsIter<'s> {
    map: &'s CartesianRoomGrid,
    id: usize,
    starting_pos: IVec2,
    edge_pos: IVec2,
    direction: IVec2,
    done: bool,
}

impl<'s> AdjacentCellsIter<'s> {
    fn new(map: &'s CartesianRoomGrid, id: usize, room_pos: IVec2) -> Self {
        // Find the room edge by moving towards +X until we aren't in the room any more
        let mut room_edge = room_pos;
        while map.cell(room_edge) == Some(&Some(id)) {
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
    type Item = IVec2;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        assert!(self.map.cell(self.edge_pos) != Some(&Some(self.id)));

        fn rotate_cw(v: IVec2) -> IVec2 {
            ivec2(-v.y, v.x)
        }
        fn rotate_ccw(v: IVec2) -> IVec2 {
            ivec2(v.y, -v.x)
        }
        // If about to hit the room, rotate counterclockwise
        if self.map.cell(self.edge_pos + self.direction) == Some(&Some(self.id)) {
            if self.map.cell(self.edge_pos + rotate_ccw(self.direction)) == Some(&Some(self.id)) {
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
        self.edge_pos += self.direction;

        // Skip outer corners
        if self.map.cell(self.edge_pos + rotate_cw(self.direction)) != Some(&Some(self.id)) {
            // #|
            // #|
            // -+
            self.direction = rotate_cw(self.direction);
            self.edge_pos += self.direction;
        }

        if self.edge_pos == self.starting_pos {
            self.done = true;
        };

        Some(result)
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

fn expand_room(map: &mut CartesianRoomGrid, rooms: impl Iterator<Item = (usize, IVec2)>) {
    const MAX_LONGEST_SIDE_EDGE: usize = 7;

    let mut expandable: Vec<(usize, RoomExpansionPositions)> = rooms
        .map(|(id, pos)| {
            map.set_cell(pos, Some(id));
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
                if positions_to_expand_to.clone().all(|pos| {
                    map.cell(pos)
                        .copied()
                        .flatten()
                        .map(|cell| cell == id)
                        .unwrap_or(true)
                }) {
                    positions_to_expand_to.for_each(|pos| map.set_cell(pos, Some(id)));
                    positions.nw_corner.x -= 1;
                } else {
                    positions.expand_left = false;
                }
            }
            if positions.expand_right {
                let positions_to_expand_to = (positions.nw_corner.y..=positions.se_corner.y)
                    .map(|y| ivec2(positions.se_corner.x + 1, y));
                if positions_to_expand_to.clone().all(|pos| {
                    map.cell(pos)
                        .copied()
                        .flatten()
                        .map(|cell| cell == id)
                        .unwrap_or(true)
                }) {
                    positions_to_expand_to.for_each(|pos| map.set_cell(pos, Some(id)));
                    positions.se_corner.x += 1;
                } else {
                    positions.expand_right = false;
                }
            }
            if positions.expand_up {
                let positions_to_expand_to = (positions.nw_corner.x..=positions.se_corner.x)
                    .map(|x| ivec2(x, positions.nw_corner.y - 1));
                if positions_to_expand_to.clone().all(|pos| {
                    map.cell(pos)
                        .copied()
                        .flatten()
                        .map(|cell| cell == id)
                        .unwrap_or(true)
                }) {
                    positions_to_expand_to.for_each(|pos| map.set_cell(pos, Some(id)));
                    positions.nw_corner.y -= 1;
                } else {
                    positions.expand_up = false;
                }
            }
            if positions.expand_down {
                let positions_to_expand_to = (positions.nw_corner.x..=positions.se_corner.x)
                    .map(|x| ivec2(x, positions.se_corner.y + 1));
                if positions_to_expand_to.clone().all(|pos| {
                    map.cell(pos)
                        .copied()
                        .flatten()
                        .map(|cell| cell == id)
                        .unwrap_or(true)
                }) {
                    positions_to_expand_to.for_each(|pos| map.set_cell(pos, Some(id)));
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

bitflags! {
    #[derive(Default)]
    pub struct Wall: u8 {
        const North = 1 << 0;
        const East = 1 << 1;
        const West = 1 << 2;
        const South = 1 << 3;
        const NorthWestCorner = 1 << 4;
        const NorthEastCorner = 1 << 5;
        const SouthWestCorner = 1 << 6;
        const SouthEastCorner = 1 << 7;
    }
}

pub fn generate_wall_map(room_map: &CartesianRoomGrid) -> CartesianGrid<Wall> {
    let mut wall_map = CartesianGrid::default();
    for (pos, &cell) in room_map.cells() {
        if let Some(cell) = cell {
            let directions = [
                ivec2(0, -1),
                ivec2(1, 0),
                ivec2(-1, 0),
                ivec2(0, 1),
                ivec2(-1, -1),
                ivec2(1, -1),
                ivec2(-1, 1),
                ivec2(1, 1),
            ];
            let wall = directions
                .into_iter()
                .map(|offset| pos + offset)
                .enumerate()
                .fold(Wall::empty(), |wall, (idx, pos)| {
                    if room_map.cell(pos) != Some(&Some(cell)) {
                        wall | Wall::from_bits_truncate(1 << idx)
                    } else {
                        wall
                    }
                });

            wall_map.set_cell(pos, wall);
        }
    }
    wall_map
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
    wall_map: &CartesianGrid<Wall>,
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

#[test]
#[cfg(test)]
fn test_doors() {
    let mut map = CartesianRoomGrid::default();
    map.set_cell(ivec2(0, 0), Some(0));
    map.set_cell(ivec2(1, 0), Some(1));
    generate_doors(
        &[ivec2(0, 0), ivec2(1, 0)],
        &map,
        &Default::default(),
        &[
            Node {
                children: vec![1],
                parent: None,
                path: Default::default(),
            },
            Node {
                children: vec![],
                parent: Some(0),
                path: Default::default(),
            },
        ],
    );
}
