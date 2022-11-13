use std::{
    collections::HashMap,
    fs,
    ops::ControlFlow,
    path::{Path, PathBuf},
};

use emath::{vec2, Pos2};
use rand::Rng;

// TODO: Custom result type
pub type Result<T> = anyhow::Result<T>;

pub struct Room {
    pub path: PathBuf,
    /// Counter-clockwise points
    pub mesh: Vec<emath::Pos2>,
    pub children: Vec<Room>,
}

const CORRIDOR_WIDTH: f32 = 100.;
const MIN_ROOM_DEPTH: f32 = 100.;

pub fn generate(path: &Path, top_left: impl Into<mint::Point2<f32>>) -> Result<Room> {
    generate_inner(path, top_left.into().into()).map(|(x, _)| x)
}

fn generate_inner(path: &Path, top_left: emath::Pos2) -> Result<(Room, f32)> {
    let mut children = vec![];
    let mut bottom_right = top_left + vec2(CORRIDOR_WIDTH, MIN_ROOM_DEPTH);
    let next_child_top_left = |bottom_right: emath::Pos2| bottom_right - vec2(0., MIN_ROOM_DEPTH);

    for dir_entry in fs::read_dir(path)? {
        let dir_entry = dir_entry?;
        if dir_entry.file_type()?.is_dir() {
            let (room, y) = generate_inner(&dir_entry.path(), next_child_top_left(bottom_right))?;
            children.push(room);
            bottom_right.y = y + MIN_ROOM_DEPTH;
        }
    }

    Ok((
        Room {
            path: path.to_owned(),
            children,
            mesh: rect_to_mesh(emath::Rect::from_two_pos(top_left, bottom_right)),
        },
        bottom_right.y,
    ))
}

pub struct Node {
    pos: emath::Pos2,
    path: PathBuf,
    parent: Option<PathBuf>,
    children: Vec<PathBuf>,
}

impl Node {
    pub fn pos(&self) -> mint::Point2<f32> {
        self.pos.into()
    }
}

pub fn generate_nodes(path: &Path) -> Result<Vec<Node>> {
    let mut nodes = Vec::<Node>::with_capacity(100);
    let mut rng = rand::thread_rng();

    let mut to_process = vec![Node {
        pos: emath::Pos2::ZERO,
        path: path.to_owned(),
        parent: None,
        children: vec![],
    }];

    while let Some(mut node_being_processed) = to_process.pop() {
        for dir_entry in fs::read_dir(&node_being_processed.path)? {
            let dir_entry = dir_entry?;
            if dir_entry.file_type()?.is_dir() {
                node_being_processed.children.push(dir_entry.path());
                to_process.push(Node {
                    pos: emath::pos2(rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0)),
                    path: dir_entry.path(),
                    parent: Some(node_being_processed.path.clone()),
                    children: vec![],
                });
            }
        }
        nodes.push(node_being_processed);
        if to_process.len() % 1000 == 0 {
            log::info!("Left to process: {}", to_process.len());
        }
    }

    Ok(nodes)
}

pub struct ForceDirectedDrawing<'nodes> {
    nodes: &'nodes mut Vec<Node>,

    t: f32,
}

impl<'nodes> ForceDirectedDrawing<'nodes> {
    const C: f32 = 0.01;
    const K: f32 = 20.;
    const T_0: f32 = Self::K;
    const TOL: f32 = 0.001;

    pub fn new(nodes: &'nodes mut Vec<Node>) -> Self {
        Self {
            nodes,
            t: Self::T_0,
        }
    }

    pub fn iterate(&mut self) -> ControlFlow<(), f32> {
        // Force-Directed Drawing Algorithm
        // https://cs.brown.edu/people/rtamassi/gdhandbook/chapters/force-directed.pdf (12.8)

        const C: f32 = ForceDirectedDrawing::C;
        const K: f32 = ForceDirectedDrawing::K;
        const TOL: f32 = ForceDirectedDrawing::TOL;

        fn f_g(x: f32, w: f32) -> f32 {
            -C * w * K * K / x
        }
        fn f_l(x: f32, d: f32, w: f32) -> f32 {
            (x - K) / d - f_g(x, w)
        }
        fn cool(t: f32) -> f32 {
            t * 0.95
        }

        let mut converged = true;
        let mut loss = 0.;

        let oldposn = self.nodes.iter().map(|n| n.pos).collect::<Vec<_>>();

        for v in 0..self.nodes.len() {
            let v_node = &self.nodes[v];
            let mut d = emath::Vec2::ZERO;
            // Calculate global (repulsive) forces
            for u in 0..self.nodes.len() {
                if u == v {
                    continue;
                }
                let delta = self.nodes[u].pos - self.nodes[v].pos;
                if delta.length() <= 0. {
                    continue;
                }
                d += delta.normalized()
                    * f_g(delta.length(), self.nodes[u].children.len() as f32 + 1.);
            }
            // Calculate local (spring) forces
            for u in 0..self.nodes.len() {
                let u_node = &self.nodes[u];
                // Only match connected nodes
                if u == v
                    || !(matches!(&u_node.parent, Some(path) if path == &v_node.path)
                        || u_node.children.contains(&v_node.path))
                {
                    continue;
                }

                let delta = self.nodes[u].pos - self.nodes[v].pos;
                if delta.length() <= 0. {
                    continue;
                }
                d += delta.normalized()
                    * f_l(
                        delta.length(),
                        v_node.children.len() as f32 + 1.,
                        u_node.children.len() as f32 + 1.,
                    );
            }
            // Reposition v
            self.nodes[v].pos = self.nodes[v].pos + d.normalized() * self.t.min(d.length());
            let delta = self.nodes[v].pos - oldposn[v];
            if delta.length() > K * TOL {
                loss += delta.length() - K * TOL;
                converged = false;
            }
        }

        self.t = cool(self.t);

        if converged {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(loss)
        }
    }

    pub fn nodes(&self) -> &Vec<Node> {
        self.nodes
    }

    pub fn nodes_mut(&mut self) -> &mut Vec<Node> {
        self.nodes
    }

    pub fn t(&self) -> f32 {
        self.t
    }
}

pub fn nodes_to_room(nodes: Vec<Node>, root: &Path) -> Room {
    let mut map: HashMap<PathBuf, Node> = HashMap::with_capacity(nodes.len());
    for node in nodes {
        map.insert(node.path.clone(), node);
    }

    fn build(map: &HashMap<PathBuf, Node>, path: &Path) -> Room {
        let node = &map[path];
        Room {
            path: node.path.clone(),
            children: node.children.iter().map(|conn| build(map, conn)).collect(),
            mesh: vec![node.pos],
        }
    }

    build(&map, root)
}

pub fn generate_v2(path: &Path) -> Result<Room> {
    log::info!("Creating nodes");

    let mut nodes = generate_nodes(path)?;

    let mut fdd = ForceDirectedDrawing::new(&mut nodes);

    loop {
        match fdd.iterate() {
            ControlFlow::Break(_) => break,
            ControlFlow::Continue(loss) => log::info!("T: {}, Loss: {}", fdd.t(), loss),
        }
    }

    log::info!("Converting nodes to rooms");

    Ok(nodes_to_room(nodes, path))
}

fn rect_to_mesh(rect: emath::Rect) -> Vec<emath::Pos2> {
    vec![
        rect.right_bottom(),
        rect.right_top(),
        rect.left_top(),
        rect.left_bottom(),
    ]
}
