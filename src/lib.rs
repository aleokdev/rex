pub mod grid;

use std::{
    fs,
    ops::ControlFlow,
    path::{Path, PathBuf},
};

use glam::{ivec2, vec2, IVec2, Vec2};
use grid::RoomTemplate;
use rand::{seq::SliceRandom, Rng};

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
