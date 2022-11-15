pub mod grid;

use std::{
    fs,
    ops::ControlFlow,
    path::{Path, PathBuf},
};

use glam::{vec2, IVec2, Vec2};

// TODO: Custom result type
pub type Result<T> = anyhow::Result<T>;

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
}

impl V3 {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self {
            nodes,
            grid: Default::default(),
            rooms: vec![],
            // We start on the root node
            stack: vec![0],
        }
    }

    pub fn iterate(&mut self) -> ControlFlow<(), ()> {}
}
