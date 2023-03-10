use std::{collections::VecDeque, path::PathBuf};

use ahash::{HashMap, HashSet};
use glam::{uvec2, uvec3, IVec3, UVec2, UVec3, Vec2};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use crate::node::Node;

#[derive(Serialize, Deserialize)]
pub struct TileMap3DChunk<Cell> {
    #[serde(with = "BigArray")]
    #[serde(bound = "Cell: Serialize + for<'ds> Deserialize<'ds>")]
    cells: [Cell; 8 * 8 * 8], // TODO: Use Self::CELL_COUNT when Rust stabilizes using generic Self types in anonymous constants
}

impl<Cell: Default + Copy> Default for TileMap3DChunk<Cell> {
    fn default() -> Self {
        Self {
            cells: [Default::default(); 8 * 8 * 8],
        }
    }
}

impl<Cell> TileMap3DChunk<Cell> {
    /// Size of the chunk in cells
    pub const CELLS_PER_AXIS: u32 = 8;
    /// Total cells in the chunk
    pub const CELL_COUNT: u32 = Self::CELLS_PER_AXIS * Self::CELLS_PER_AXIS * Self::CELLS_PER_AXIS;

    pub fn cell(&self, pos: UVec3) -> Option<&Cell> {
        let UVec3 { x, y, z } = pos;

        self.contains(pos)
            .then(|| {
                self.cells.get(
                    (x + y * Self::CELLS_PER_AXIS + z * Self::CELLS_PER_AXIS * Self::CELLS_PER_AXIS)
                        as usize,
                )
            })
            .flatten()
    }

    pub fn set_cell(&mut self, pos: UVec3, value: Cell) {
        let UVec3 { x, y, z } = pos;
        assert!(
            self.contains(pos),
            "Tried to set node of wire grid which wasn't contained within"
        );
        self.cells[(x + y * Self::CELLS_PER_AXIS + z * Self::CELLS_PER_AXIS * Self::CELLS_PER_AXIS)
            as usize] = value;
    }

    pub fn contains(&self, pos: UVec3) -> bool {
        let UVec3 { x, y, z } = pos.into();
        x < Self::CELLS_PER_AXIS && y < Self::CELLS_PER_AXIS && z < Self::CELLS_PER_AXIS
    }

    pub fn cells(&self) -> impl Iterator<Item = (UVec3, &Cell)> {
        self.cells.iter().enumerate().map(|(i, cell)| {
            (
                uvec3(
                    i as u32 % Self::CELLS_PER_AXIS,
                    (i as u32 / Self::CELLS_PER_AXIS) % Self::CELLS_PER_AXIS,
                    i as u32 / Self::CELLS_PER_AXIS / Self::CELLS_PER_AXIS,
                ),
                cell,
            )
        })
    }
}

#[derive(Default)]
pub struct TileMap3D<Cell: Clone> {
    chunks: HashMap<IVec3, TileMap3DChunk<Cell>>,
}

type PieceSetId = String;

pub struct Rect2D<T> {
    pub x: T,
    pub y: T,
    pub w: T,
    pub h: T,
}

pub struct Image {
    path: PathBuf,
    source: Rect2D<u32>,
    target: Rect2D<f32>,
}

pub struct Piece {
    name: String,
    sets: HashSet<PieceSetId>,
    image: Image,
}

// we first do a piece pass with a large grid where one cell is as wide and long as a wall but as high as a floor.
//
pub struct V5 {
    nodes: Vec<Node>,
    /// Indicates the nodes that the algorithm has yet to process.
    /// They will be processed from front to back, adding new items to the back.
    queue: VecDeque<usize>,

    pub map: TileMap3D<()>,
}

impl V5 {
    pub fn new(nodes: Vec<Node>) -> Self {
        let mut queue = VecDeque::new();
        queue.push_back(0);

        Self {
            nodes,
            queue,
            map: Default::default(),
        }
    }

    pub fn step(&mut self) {}
}
