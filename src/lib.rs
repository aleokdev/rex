use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use emath::vec2;
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

pub fn generate_v2(path: &Path) -> Result<Room> {
    struct Node {
        pos: emath::Pos2,
        path: PathBuf,
        parent: Option<PathBuf>,
        children: Vec<PathBuf>,
    }

    let mut nodes = Vec::<Node>::new();
    let mut rng = rand::thread_rng();

    log::info!("Creating nodes");

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

    // Force-Directed Drawing Algorithm
    // https://cs.brown.edu/people/rtamassi/gdhandbook/chapters/force-directed.pdf (12.8)
    const C: f32 = 1.;
    const K: f32 = 3.;
    const T_0: f32 = 100.;
    const TOL: f32 = 0.1;

    fn f_g(x: f32, w: f32) -> f32 {
        -C * w * K * K / x
    }
    fn f_l(x: f32, d: f32, w: f32) -> f32 {
        (x - K) / d - f_g(x, w)
    }
    fn cool(t: f32) -> f32 {
        t * 9. / 10.
    }

    let mut t = T_0;
    let mut newposn = nodes.iter().map(|node| node.pos).collect::<Vec<_>>();

    loop {
        let mut converged = true;
        let mut loss = 0.;

        for v in 0..nodes.len() {
            let mut d = emath::Vec2::ZERO;
            // Calculate global (repulsive) forces
            for u in 0..nodes.len() {
                if u == v {
                    continue;
                }
                let delta = newposn[u] - newposn[v];
                if delta.length() <= 0. {
                    continue;
                }
                d += delta.normalized() * f_g(delta.length(), newposn[u].to_vec2().length());
            }
            // Calculate local (spring) forces
            for u in 0..nodes.len() {
                if u == v {
                    continue;
                }
                if (newposn[u] - newposn[v]).length() > 50. {
                    continue;
                }

                let delta = newposn[u] - newposn[v];
                if delta.length() <= 0. {
                    continue;
                }
                d += delta.normalized()
                    * f_l(
                        delta.length(),
                        newposn[v].to_vec2().length().max(0.01),
                        newposn[u].to_vec2().length(),
                    );
            }
            // Reposition v
            let newpos = newposn[v] + d.normalized() * t.min(d.length());
            let delta = newpos - newposn[v];
            newposn[v] = newpos;
            if delta.length() > K * TOL {
                loss += delta.length() - K * TOL;
                converged = false;
            }
        }

        log::info!("A");
        log::info!("T: {}, Loss: {}", t, loss);
        t = cool(t);

        if converged {
            break;
        }
    }

    nodes
        .iter_mut()
        .zip(newposn.into_iter())
        .for_each(|(node, npos)| node.pos = npos);

    log::info!("Converting nodes");

    let mut map: HashMap<PathBuf, Node> = HashMap::new();
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

    Ok(build(&map, path))
}

fn rect_to_mesh(rect: emath::Rect) -> Vec<emath::Pos2> {
    vec![
        rect.right_bottom(),
        rect.right_top(),
        rect.left_top(),
        rect.left_bottom(),
    ]
}
