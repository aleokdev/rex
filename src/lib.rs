mod space;
pub mod wfc;

use std::{
    fs,
    ops::ControlFlow,
    path::{Path, PathBuf},
    sync::{atomic::AtomicUsize, Arc},
};

use glam::{ivec2, IVec2, Vec2};
use rand::{seq::SliceRandom, Rng};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use space::Space;

// TODO: Custom result type
pub type Result<T> = anyhow::Result<T>;

/// A Rex Room. This represents a directory in 3D space.
#[derive(Default)]
pub struct Room {
    pub path: PathBuf,
    /// Points in non-specified winding indicating the contour of the floor mesh.
    pub mesh: Vec<mint::Vector2<f32>>,
    // TODO: How do we represent connections / doors / doorways?
    // Maybe with points & normals, then add doorways in walls in the vertex shader?
    // pub connections: Vec<Connection>,
}

/// Used to refer to a room. This is the index to a room in the rooms vector.
pub type RoomId = usize;

/// A node in the directory tree, specifying its path, parent and children.
// TODO: Specify shortcut behavior
#[derive(Clone, Debug)]
pub struct Node {
    path: PathBuf,
    parent: Option<usize>,
    children: Vec<usize>,
}

/// Returns a node and its children from a directory path.
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

/// Algorithm that tries to place connected rooms close together, without letting them intersect,
/// with a bit of space in between them.
pub struct RoomPlacer {
    nodes: Vec<Node>,
    rooms: Vec<Room>,
    /// Indicates the node branch the algorithm has gone through.
    /// It will unwind when there are no more children to process in a given node,
    /// and increase its size when a node has children to process.
    stack: Vec<usize>,
    space: Space,
}

impl RoomPlacer {
    #[inline]
    pub fn new(nodes: Vec<Node>) -> Self {
        let mut space = Space::default();
        // Allocate root node
        space.allocate_at(0, Vec2::ZERO, 1.).unwrap();
        let mut rooms: Vec<Room> = nodes
            .iter()
            .map(|node| Room {
                path: node.path.clone(),
                ..Default::default()
            })
            .collect();
        // Generate root room mesh
        rooms[0].mesh = Self::generate_room_mesh(4, 1., Vec2::ZERO);

        Self {
            rooms,
            nodes,
            // We start processing the root node
            stack: vec![0],
            space: space,
        }
    }

    pub fn iterate(&mut self, rng: &mut impl Rng) -> ControlFlow<(), ()> {
        let Some(node_idx) = self.stack.pop() else { return ControlFlow::Break(()) };
        let node = &self.nodes[node_idx];

        fn radius_fn(_child_count: usize) -> f32 {
            1.
        }

        // Breadth-first rather than depth-first to encourage having connected rooms next to each other
        for &child_idx in node.children.iter() {
            let child_node = &self.nodes[child_idx];
            let radius = radius_fn(child_node.children.len());
            let parent_radius = radius_fn(node.children.len());

            let final_pos = self.space.allocate_near(
                child_idx,
                Vec2::from(*self.rooms[node_idx].mesh.choose(rng).unwrap()),
                parent_radius * 2.,
                radius * 2.,
                rng,
            );

            let point_count = rng.gen_range(4..=6);

            let points = Self::generate_room_mesh(point_count, radius, final_pos);
            self.rooms[child_idx].mesh = points;

            self.stack.push(child_idx);
        }

        if self.stack.is_empty() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    fn generate_room_mesh(
        point_count: usize,
        radius: f32,
        position: Vec2,
    ) -> Vec<mint::Vector2<f32>> {
        (0..point_count)
            .map(|point_idx| {
                std::f32::consts::FRAC_PI_2
                    + (point_idx as f32 / point_count as f32) * std::f32::consts::TAU
            })
            .map(|angle| (Vec2::from(angle.sin_cos()) * radius + position).into())
            .collect()
    }

    #[inline]
    pub fn nodes(&self) -> &[Node] {
        self.nodes.as_ref()
    }

    #[inline]
    pub fn rooms(&self) -> &[Room] {
        self.rooms.as_ref()
    }

    #[inline]
    pub fn build(self) -> (Vec<Node>, Vec<Room>, Space) {
        (self.nodes, self.rooms, self.space)
    }
}

/// Algorithm that places corridors in between the rooms given, and returns the resulting paths.
/// These paths may need smoothing & simplification via [`CorridorSimplifier`].
pub struct CorridorPlacer {
    nodes: Vec<Node>,
    rooms: Vec<Room>,
    paths: Vec<Vec<mint::Vector2<f32>>>,
    space: Space,
    current_node: usize,
}

impl CorridorPlacer {
    #[inline]
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
        let room_mesh_first_point = Vec2::from(room.mesh[0]);
        // Connect this node with its children
        for &child_idx in &node.children {
            let child_room = &self.rooms[child_idx];
            let target = child_room
                .mesh
                .iter()
                .map(|&point| Vec2::from(point))
                .min_by(|&point_a, &point_b| {
                    (room_mesh_first_point - point_a)
                        .length_squared()
                        .total_cmp(&(room_mesh_first_point - point_b).length_squared())
                })
                .unwrap();
            let start = room
                .mesh
                .iter()
                .map(|&point| Vec2::from(point))
                .min_by(|&point_a, &point_b| {
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
            let target_astar = world_units_to_astar_point(target);
            let success = |&point: &IVec2| {
                let diff = point - target_astar;
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

    #[inline]
    pub fn paths(&self) -> &[Vec<mint::Vector2<f32>>] {
        self.paths.as_ref()
    }

    #[inline]
    pub fn nodes(&self) -> &[Node] {
        self.nodes.as_ref()
    }

    #[inline]
    pub fn rooms(&self) -> &[Room] {
        self.rooms.as_ref()
    }

    #[inline]
    pub fn build(self) -> (Vec<Node>, Vec<Room>, Vec<Vec<mint::Vector2<f32>>>, Space) {
        (self.nodes, self.rooms, self.paths, self.space)
    }
}

/// Algorithm that simplifies and cleans up the paths given.
pub struct CorridorSimplifier {
    paths: Vec<Vec<mint::Vector2<f32>>>,
    space: Space,
}

impl CorridorSimplifier {
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
        let converged = Arc::new(AtomicUsize::from(self.paths.len()));
        self.paths.par_iter_mut().for_each(|path| {
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
                converged.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            }
        });
        // Finish if 95% of paths have converged
        if converged.load(std::sync::atomic::Ordering::SeqCst) >= self.paths.len() * 95 / 100 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    pub fn paths(&self) -> &[Vec<mint::Vector2<f32>>] {
        self.paths.as_ref()
    }
}
