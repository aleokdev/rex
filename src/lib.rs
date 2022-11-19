pub mod grid;
mod space;

use std::{
    fs,
    ops::ControlFlow,
    path::{Path, PathBuf},
};

use glam::{ivec2, vec2, IVec2, Vec2};
use grid::RoomTemplate;
use rand::{seq::SliceRandom, Rng};
use space::Space;

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
    path: PathBuf,
    parent: Option<usize>,
    children: Vec<usize>,
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

/// To-be-named algorithm using an infinite cell grid, trying to place rooms as close as possible but allowing placing corridors as connections
/// that intersect each other.
pub struct V3 {
    nodes: Vec<Node>,
    grid: grid::CartesianRoomGrid,
    rooms: Vec<Room>,
    /// Indicates the node branch the algorithm has gone through.
    /// It will unwind when there are no more children to process in a given node,
    /// and increase its size when a node has children to process.
    stack: Vec<usize>,
    template_db: RoomTemplateDb,
}

impl V3 {
    pub fn new(nodes: Vec<Node>, template_db: RoomTemplateDb) -> Self {
        Self {
            rooms: nodes
                .iter()
                .map(|node| Room {
                    path: node.path.clone(),
                    ..Default::default()
                })
                .collect(),
            nodes,
            grid: Default::default(),
            // We start on the root node
            stack: vec![0],
            template_db,
        }
    }

    pub fn iterate(&mut self, rng: &mut impl Rng) -> ControlFlow<(), ()> {
        let Some(node_idx) = self.stack.pop() else { return ControlFlow::Break(()) };
        let node = &self.nodes[node_idx];

        let template_to_place = self.template_db.random(rng).unwrap();

        let final_pos = self.grid.place_template_near(
            template_to_place,
            node_idx,
            node.parent
                .map(|parent| *self.rooms[parent].mesh.choose(rng).unwrap())
                .unwrap_or(Vec2::ZERO),
            rng,
        );

        self.rooms[node_idx].mesh = template_to_place
            .outline()
            .iter()
            .map(|&v| v + final_pos.as_vec2())
            .collect();

        for &child in &node.children {
            self.stack.push(child);
        }

        if self.stack.is_empty() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    pub fn nodes(&self) -> &[Node] {
        self.nodes.as_ref()
    }

    pub fn rooms(&self) -> &[Room] {
        self.rooms.as_ref()
    }
}

/// To-be-named algorithm using an infinite space using allocations to reserve space for rooms.
pub struct V4 {
    nodes: Vec<Node>,
    rooms: Vec<Room>,
    /// Indicates the node branch the algorithm has gone through.
    /// It will unwind when there are no more children to process in a given node,
    /// and increase its size when a node has children to process.
    stack: Vec<usize>,
    space: Space,
}

impl V4 {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self {
            rooms: nodes
                .iter()
                .map(|node| Room {
                    path: node.path.clone(),
                    ..Default::default()
                })
                .collect(),
            nodes,
            // We start on the root node
            stack: vec![0],
            space: Default::default(),
        }
    }

    pub fn iterate(&mut self, rng: &mut impl Rng) -> ControlFlow<(), ()> {
        let Some(node_idx) = self.stack.pop() else { return ControlFlow::Break(()) };
        let node = &self.nodes[node_idx];

        fn radius_fn(child_count: usize) -> f32 {
            1.
        }

        let radius = radius_fn(node.children.len());
        let parent_radius = node
            .parent
            .map(|parent_idx| radius_fn(self.nodes[parent_idx].children.len()))
            .unwrap_or(0.);

        let final_pos = self.space.allocate_near(
            node_idx,
            node.parent
                .map(|parent_idx| *self.rooms[parent_idx].mesh.choose(rng).unwrap())
                .unwrap_or(Vec2::ZERO),
            parent_radius * 2.,
            radius * 2.,
            rng,
        );

        let point_count = rng.gen_range(4..=6);

        let points = (0..point_count)
            .map(|point_idx| {
                std::f32::consts::FRAC_PI_2
                    + (point_idx as f32 / point_count as f32) * std::f32::consts::TAU
            })
            .map(|angle| Vec2::from(angle.sin_cos()) * radius + final_pos);

        self.rooms[node_idx].mesh = points.collect();

        for &child in &node.children {
            self.stack.push(child);
        }

        if self.stack.is_empty() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    pub fn nodes(&self) -> &[Node] {
        self.nodes.as_ref()
    }

    pub fn rooms(&self) -> &[Room] {
        self.rooms.as_ref()
    }

    pub fn build(self) -> (Vec<Node>, Vec<Room>, Space) {
        (self.nodes, self.rooms, self.space)
    }
}

pub struct V4CorridorSolver {
    nodes: Vec<Node>,
    rooms: Vec<Room>,
    paths: Vec<Vec<mint::Vector2<f32>>>,
    space: Space,
    current_node: usize,
}

impl V4CorridorSolver {
    pub fn new(nodes: Vec<Node>, rooms: Vec<Room>, space: Space) -> Self {
        Self {
            nodes,
            rooms,
            paths: vec![],
            space,
            current_node: 0,
        }
    }

    pub fn iterate(&mut self) -> ControlFlow<(), ()> {
        let node = &self.nodes[self.current_node];
        let room = &self.rooms[self.current_node];
        // Connect this node with its children
        for &child_idx in &node.children {
            let child_room = &self.rooms[child_idx];
            let &target = child_room
                .mesh
                .iter()
                .min_by(|&&point_a, &&point_b| {
                    (room.mesh[0] - point_a)
                        .length_squared()
                        .total_cmp(&(room.mesh[0] - point_b).length_squared())
                })
                .unwrap();
            let &start = room
                .mesh
                .iter()
                .min_by(|&&point_a, &&point_b| {
                    (target - point_a)
                        .length_squared()
                        .total_cmp(&(target - point_b).length_squared())
                })
                .unwrap();
            const RESOLUTION: f32 = 0.5;
            const MIN_TARGET_DISTANCE: f32 = RESOLUTION * RESOLUTION * std::f32::consts::SQRT_2;
            let astar_point_to_world_units = |point: IVec2| point.as_vec2() / RESOLUTION;
            let world_units_to_astar_point = |point: Vec2| (point * RESOLUTION).as_ivec2();
            let start_node = world_units_to_astar_point(start);
            let successors = |&point: &IVec2| {
                let point_and_cost = |delta: IVec2| {
                    let point = point + delta;
                    let cost = if self.space.is_point_allocated_by_any_other_than(
                        child_idx,
                        astar_point_to_world_units(point),
                    ) {
                        10
                    } else {
                        1
                    };
                    (point, cost)
                };
                [
                    point_and_cost(ivec2(0, 1)),
                    point_and_cost(ivec2(1, 0)),
                    point_and_cost(ivec2(0, -1)),
                    point_and_cost(ivec2(-1, 0)),
                ]
            };
            let heuristic = |&point: &IVec2| {
                (astar_point_to_world_units(point) - target)
                    .length()
                    .floor() as i32
            };
            let success = |&point: &IVec2| {
                let diff = point - world_units_to_astar_point(target);
                (diff.dot(diff) as f32) < MIN_TARGET_DISTANCE
            };
            let path =
                pathfinding::directed::astar::astar(&start_node, successors, heuristic, success)
                    .unwrap()
                    .0;
            let mut path: Vec<mint::Vector2<f32>> = path
                .into_iter()
                .map(|x| astar_point_to_world_units(x).into())
                .collect();
            if let Some(first) = path.first_mut() {
                *first = start.into();
            }
            if let Some(last) = path.last_mut() {
                *last = target.into();
            }
            self.paths.push(path);
        }

        self.current_node += 1;
        if self.current_node >= self.nodes.len() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    pub fn paths(&self) -> &[Vec<mint::Vector2<f32>>] {
        self.paths.as_ref()
    }

    pub fn nodes(&self) -> &[Node] {
        self.nodes.as_ref()
    }

    pub fn rooms(&self) -> &[Room] {
        self.rooms.as_ref()
    }

    pub fn build(self) -> (Vec<Node>, Vec<Room>, Vec<Vec<mint::Vector2<f32>>>, Space) {
        (self.nodes, self.rooms, self.paths, self.space)
    }
}

pub struct V4CorridorSmoother {
    paths: Vec<Vec<mint::Vector2<f32>>>,
    space: Space,
}

impl V4CorridorSmoother {
    pub fn new(paths: Vec<Vec<mint::Vector2<f32>>>, space: Space) -> Self {
        // x2 interpolation
        let paths = paths
            .into_iter()
            .map(|path| {
                path.windows(2)
                    .flat_map(|ps| {
                        (0..=4).map(|idx| {
                            Vec2::from(ps[0])
                                .lerp(Vec2::from(ps[1]), idx as f32 / 4.)
                                .into()
                        })
                    })
                    .collect()
            })
            .collect();
        Self { paths, space }
    }

    pub fn iterate(&mut self) -> ControlFlow<(), ()> {
        let mut converged = self.paths.len();
        for path in self.paths.iter_mut() {
            let mut path_converged = true;
            let path_len = path.len();
            if path_len > 2 {
                for point_idx in 1..path_len - 1 {
                    path[point_idx] = ((Vec2::from(path[point_idx - 1])
                        + Vec2::from(path[point_idx])
                        + Vec2::from(path[point_idx + 1]))
                        / 3.)
                        .into();
                    let displacement =
                        0.01 * self.space.force_displacement((path[point_idx]).into());
                    path[point_idx] = (Vec2::from(path[point_idx]) + displacement).into();
                    let displacement_length = displacement.length();
                    if displacement_length > 0.13 {
                        path_converged = false;
                    }
                }
            }
            if !path_converged {
                converged -= 1;
            }
        }
        // Finish if 95% of paths have converged
        if converged >= self.paths.len() * 95 / 100 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    pub fn paths(&self) -> &[Vec<mint::Vector2<f32>>] {
        self.paths.as_ref()
    }
}
