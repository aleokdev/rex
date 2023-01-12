pub mod grid;
pub mod space;

pub use glam;

use std::{
    collections::VecDeque,
    fs,
    ops::{ControlFlow, Range},
    path::{Path, PathBuf},
};

use ahash::HashSet;
use bitflags::bitflags;
use glam::{ivec2, vec2, IVec2, Vec2};
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
    room_positions: Vec<Vec2>,
}

impl V3 {
    pub fn new(nodes: Vec<Node>) -> Self {
        let leaf_count = nodes.iter().filter(|node| node.children.len() == 0).count();

        let mut leaf_idx = 0;
        let node_positions = nodes.iter().map(|node| {
            if node.children.len() == 0 {
                let rotation = (leaf_idx as f32 / leaf_count as f32) * std::f32::consts::TAU;
                leaf_idx += 1;
                vec2(rotation.cos(), rotation.sin()) * 250.
            } else {
                Vec2::ZERO
            }
        });
        Self {
            room_positions: node_positions.collect(),
            nodes,
        }
    }

    pub fn iterate(&mut self) -> ControlFlow<(), ()> {
        let mut new_positions = self.room_positions.clone();

        fn iter(
            nodes: &[Node],
            old_positions: &Vec<Vec2>,
            new_positions: &mut Vec<Vec2>,
            id: usize,
        ) {
            let node = &nodes[id];
            if !node.children.is_empty() {
                fn iter_sum_children(
                    nodes: &[Node],
                    positions: &Vec<Vec2>,
                    id: usize,
                ) -> (Vec2, usize) {
                    let node = &nodes[id];
                    if node.children.len() == 0 {
                        (positions[id], 1)
                    } else {
                        node.children
                            .iter()
                            .copied()
                            .map(|id| iter_sum_children(nodes, positions, id))
                            .fold((positions[id], 1), |(pos, count), (p, c)| {
                                (pos + p, count + c)
                            })
                    }
                }
                let (mut pos_sum, mut count) = iter_sum_children(nodes, old_positions, id);
                pos_sum -= old_positions[id];
                count -= 1;
                if let Some(parent) = node.parent {
                    pos_sum += old_positions[parent] * count as f32;
                    count *= 2;
                }
                new_positions[id] = pos_sum / count as f32;
                for &child in node.children.iter() {
                    iter(nodes, old_positions, new_positions, child);
                }
            }
        }

        iter(&self.nodes, &self.room_positions, &mut new_positions, 0);
        self.room_positions = new_positions;
        ControlFlow::Continue(())
    }

    pub fn nodes(&self) -> &[Node] {
        self.nodes.as_ref()
    }

    pub fn room_positions(&self) -> &[Vec2] {
        self.room_positions.as_ref()
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

pub struct V3Expand {
    grid: CartesianRoomGrid,
    expansion: Vec<RoomExpansionPositions>,
    expandable: Vec<usize>,
}
const MAX_LONGEST_SIDE_EDGE: usize = 7;

impl V3Expand {
    pub fn from_v3(v3: V3) -> V3Expand {
        let mut grid = CartesianRoomGrid::default();
        Self {
            expansion: v3
                .room_positions
                .into_iter()
                .enumerate()
                .map(|(id, pos)| {
                    let pos = pos.as_ivec2();
                    grid.set_cell(pos, Some(id));
                    RoomExpansionPositions {
                        nw_corner: pos,
                        se_corner: pos,
                        expand_down: true,
                        expand_left: true,
                        expand_right: true,
                        expand_up: true,
                        longest_side_edge: 1,
                    }
                })
                .collect(),
            grid,
            expandable: (0..v3.nodes.len()).collect(),
        }
    }

    pub fn iterate(&mut self) -> ControlFlow<(), ()> {
        self.expandable.retain(|&id| {
            let positions = &mut self.expansion[id];
            if positions.expand_left {
                let positions_to_expand_to = (positions.nw_corner.y..=positions.se_corner.y)
                    .map(|y| ivec2(positions.nw_corner.x - 1, y));
                if positions_to_expand_to.clone().all(|pos| {
                    self.grid
                        .cell(pos)
                        .copied()
                        .flatten()
                        .map(|cell| cell == id)
                        .unwrap_or(true)
                }) {
                    positions_to_expand_to.for_each(|pos| self.grid.set_cell(pos, Some(id)));
                    positions.nw_corner.x -= 1;
                } else {
                    positions.expand_left = false;
                }
            }
            if positions.expand_right {
                let positions_to_expand_to = (positions.nw_corner.y..=positions.se_corner.y)
                    .map(|y| ivec2(positions.se_corner.x + 1, y));
                if positions_to_expand_to.clone().all(|pos| {
                    self.grid
                        .cell(pos)
                        .copied()
                        .flatten()
                        .map(|cell| cell == id)
                        .unwrap_or(true)
                }) {
                    positions_to_expand_to.for_each(|pos| self.grid.set_cell(pos, Some(id)));
                    positions.se_corner.x += 1;
                } else {
                    positions.expand_right = false;
                }
            }
            if positions.expand_up {
                let positions_to_expand_to = (positions.nw_corner.x..=positions.se_corner.x)
                    .map(|x| ivec2(x, positions.nw_corner.y - 1));
                if positions_to_expand_to.clone().all(|pos| {
                    self.grid
                        .cell(pos)
                        .copied()
                        .flatten()
                        .map(|cell| cell == id)
                        .unwrap_or(true)
                }) {
                    positions_to_expand_to.for_each(|pos| self.grid.set_cell(pos, Some(id)));
                    positions.nw_corner.y -= 1;
                } else {
                    positions.expand_up = false;
                }
            }
            if positions.expand_down {
                let positions_to_expand_to = (positions.nw_corner.x..=positions.se_corner.x)
                    .map(|x| ivec2(x, positions.se_corner.y + 1));
                if positions_to_expand_to.clone().all(|pos| {
                    self.grid
                        .cell(pos)
                        .copied()
                        .flatten()
                        .map(|cell| cell == id)
                        .unwrap_or(true)
                }) {
                    positions_to_expand_to.for_each(|pos| self.grid.set_cell(pos, Some(id)));
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

        if self.expandable.is_empty() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    pub fn grid(&self) -> &CartesianRoomGrid {
        &self.grid
    }
}

pub struct V3FillGaps {
    grid: CartesianRoomGrid,
    expansion: Vec<RoomExpansionPositions>,
    index: usize,
}

impl V3FillGaps {
    pub fn from_v3_expand(mut v3: V3Expand) -> V3FillGaps {
        for exp in v3.expansion.iter_mut() {
            exp.expand_down = true;
            exp.expand_left = true;
            exp.expand_right = true;
            exp.expand_up = true;
        }
        Self {
            grid: v3.grid,
            expansion: v3.expansion,
            index: 0,
        }
    }
    pub fn iterate(&mut self) -> ControlFlow<(), ()> {
        if let Some(positions) = self.expansion.get_mut(self.index) {
            // Keep going while we can expand
            while positions.longest_side_edge < MAX_LONGEST_SIDE_EDGE
                && (positions.expand_down
                    || positions.expand_left
                    || positions.expand_right
                    || positions.expand_up)
            {
                // Keep expanding while there are empty cells to expand to
                if positions.expand_left {
                    let positions_to_expand_to = (positions.nw_corner.y..=positions.se_corner.y)
                        .map(|y| ivec2(positions.nw_corner.x - 1, y))
                        .filter(|&pos| self.grid.cell(pos).copied().flatten().is_none())
                        .collect::<Vec<_>>();
                    if positions_to_expand_to.len() != 0 {
                        positions_to_expand_to
                            .into_iter()
                            .for_each(|pos| self.grid.set_cell(pos, Some(self.index)));
                        positions.nw_corner.x -= 1;
                    } else {
                        positions.expand_left = false;
                    }
                }
                if positions.expand_right {
                    let positions_to_expand_to = (positions.nw_corner.y..=positions.se_corner.y)
                        .map(|y| ivec2(positions.se_corner.x + 1, y))
                        .filter(|&pos| self.grid.cell(pos).copied().flatten().is_none())
                        .collect::<Vec<_>>();
                    if positions_to_expand_to.len() != 0 {
                        positions_to_expand_to
                            .into_iter()
                            .for_each(|pos| self.grid.set_cell(pos, Some(self.index)));
                        positions.se_corner.x += 1;
                    } else {
                        positions.expand_right = false;
                    }
                }

                if positions.expand_up {
                    let positions_to_expand_to = (positions.nw_corner.x..=positions.se_corner.x)
                        .map(|x| ivec2(x, positions.nw_corner.y - 1))
                        .filter(|&pos| self.grid.cell(pos).copied().flatten().is_none())
                        .collect::<Vec<_>>();
                    if positions_to_expand_to.len() != 0 {
                        positions_to_expand_to
                            .into_iter()
                            .for_each(|pos| self.grid.set_cell(pos, Some(self.index)));
                        positions.nw_corner.y -= 1;
                    } else {
                        positions.expand_up = false;
                    }
                }

                if positions.expand_down {
                    let positions_to_expand_to = (positions.nw_corner.x..=positions.se_corner.x)
                        .map(|x| ivec2(x, positions.se_corner.y + 1))
                        .filter(|&pos| self.grid.cell(pos).copied().flatten().is_none())
                        .collect::<Vec<_>>();
                    if positions_to_expand_to.len() != 0 {
                        positions_to_expand_to
                            .into_iter()
                            .for_each(|pos| self.grid.set_cell(pos, Some(self.index)));
                        positions.se_corner.y += 1;
                    } else {
                        positions.expand_down = false;
                    }
                }

                positions.longest_side_edge += 1;
            }

            self.index += 1;

            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    }

    pub fn grid(&self) -> &CartesianRoomGrid {
        &self.grid
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

            if room_edge != starting_pos {
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
