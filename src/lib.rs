use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use emath::vec2;

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

fn rect_to_mesh(rect: emath::Rect) -> Vec<emath::Pos2> {
    vec![
        rect.right_bottom(),
        rect.right_top(),
        rect.left_top(),
        rect.left_bottom(),
    ]
}
