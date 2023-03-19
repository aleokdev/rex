use std::{collections::VecDeque, num::NonZeroUsize, path::PathBuf};

use ahash::{HashMap, HashSet};
use glam::{ivec3, uvec2, uvec3, IVec3, UVec2, UVec3, Vec2};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use crate::{math::floordiv, node::Node};

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

pub struct TileMap3D<Cell> {
    chunks: HashMap<IVec3, TileMap3DChunk<Cell>>,
}

// For some reason cannot derive automatically
impl<Cell> Default for TileMap3D<Cell> {
    fn default() -> Self {
        Self {
            chunks: Default::default(),
        }
    }
}

impl<Cell: Default + Copy> TileMap3D<Cell> {
    pub fn cell(&self, pos: IVec3) -> Option<&Cell> {
        let (chunk_pos, local_pos) = Self::global_coords_to_chunk_and_local_coords(pos);

        self.cell_at(chunk_pos, local_pos)
    }

    pub fn cell_at(&self, chunk_pos: IVec3, local_pos: UVec3) -> Option<&Cell> {
        self.chunks.get(&chunk_pos).map(|chunk| {
            chunk
                .cell(local_pos)
                .expect("Cell calculations aren't correct")
        })
    }

    pub fn set_cell(&mut self, pos: IVec3, value: Cell) {
        let (chunk_pos, local_pos) = Self::global_coords_to_chunk_and_local_coords(pos);

        self.set_cell_at(chunk_pos, local_pos, value)
    }

    pub fn set_cell_at(&mut self, chunk_pos: IVec3, local_pos: UVec3, value: Cell) {
        self.chunks
            .entry(chunk_pos)
            .or_insert_with(|| Default::default())
            .set_cell(local_pos, value)
    }

    pub fn global_coords_to_chunk_and_local_coords(vec: IVec3) -> (IVec3, UVec3) {
        let IVec3 { x, y, z } = vec;
        let x_c = floordiv(x, TileMap3DChunk::<Cell>::CELLS_PER_AXIS as i32);
        let x_local = x - x_c * TileMap3DChunk::<Cell>::CELLS_PER_AXIS as i32;
        let y_c = floordiv(y, TileMap3DChunk::<Cell>::CELLS_PER_AXIS as i32);
        let y_local = y - y_c * TileMap3DChunk::<Cell>::CELLS_PER_AXIS as i32;
        let z_c = floordiv(z, TileMap3DChunk::<Cell>::CELLS_PER_AXIS as i32);
        let z_local = z - z_c * TileMap3DChunk::<Cell>::CELLS_PER_AXIS as i32;
        assert!(x_local >= 0 && x_local < TileMap3DChunk::<Cell>::CELLS_PER_AXIS as i32);
        assert!(y_local >= 0 && y_local < TileMap3DChunk::<Cell>::CELLS_PER_AXIS as i32);
        assert!(z_local >= 0 && z_local < TileMap3DChunk::<Cell>::CELLS_PER_AXIS as i32);
        (
            ivec3(x_c, y_c, z_c),
            uvec3(x_local as u32, y_local as u32, z_local as u32),
        )
    }

    pub fn chunk_and_local_coords_to_global_coords(chunk_pos: IVec3, local_pos: UVec3) -> IVec3 {
        chunk_pos * TileMap3DChunk::<Cell>::CELLS_PER_AXIS as i32 + local_pos.as_ivec3()
    }

    pub fn cells(&self) -> impl Iterator<Item = (IVec3, &Cell)> {
        self.chunks.iter().flat_map(|(&chunk_pos, chunk)| {
            chunk.cells().map(move |(local_pos, cell)| {
                (
                    Self::chunk_and_local_coords_to_global_coords(chunk_pos, local_pos).into(),
                    cell,
                )
            })
        })
    }
}

type PieceSetId = String;

pub struct Rect2D<T> {
    pub x: T,
    pub y: T,

    /// Length of the rect in the X axis, towards +X.
    pub w: T,
    /// Length of the rect in the Y axis, towards +Y.
    pub h: T,
}

pub struct Rect3D<T> {
    pub x: T,
    pub y: T,
    pub z: T,

    /// Length of the rect in the X axis, towards +X.
    pub w: T,
    /// Length of the rect in the Y axis, towards +Y.
    pub d: T,
    /// Length of the rect in the Z axis, towards +Z.
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

#[derive(Clone)]
pub struct DirectionMap<T> {
    values: [T; 6],
}

impl<T> DirectionMap<T> {
    pub fn get(&self, dir: Direction) -> &T {
        match dir {
            Direction::Up => &self.values[0],
            Direction::Down => &self.values[1],
            Direction::East => &self.values[2],
            Direction::West => &self.values[3],
            Direction::North => &self.values[4],
            Direction::South => &self.values[5],
        }
    }

    pub fn get_mut(&mut self, dir: Direction) -> &mut T {
        match dir {
            Direction::Up => &mut self.values[0],
            Direction::Down => &mut self.values[1],
            Direction::East => &mut self.values[2],
            Direction::West => &mut self.values[3],
            Direction::North => &mut self.values[4],
            Direction::South => &mut self.values[5],
        }
    }

    pub fn set(&mut self, dir: Direction, val: T) {
        *self.get_mut(dir) = val;
    }
}

impl<T: Clone> DirectionMap<T> {
    pub fn new_with_all(t: T) -> Self {
        DirectionMap {
            values: [
                t.clone(),
                t.clone(),
                t.clone(),
                t.clone(),
                t.clone(),
                t.clone(),
            ],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
    East,
    West,
    North,
    South,
}

impl Direction {
    fn inverse(self) -> Direction {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::East => Direction::West,
            Direction::West => Direction::East,
            Direction::North => Direction::South,
            Direction::South => Direction::North,
        }
    }
}

impl From<Direction> for glam::IVec3 {
    fn from(value: Direction) -> Self {
        match value {
            Direction::Up => glam::IVec3::Z,
            Direction::Down => glam::IVec3::NEG_Z,
            Direction::East => glam::IVec3::X,
            Direction::West => glam::IVec3::NEG_X,
            Direction::North => glam::IVec3::Y,
            Direction::South => glam::IVec3::NEG_Y,
        }
    }
}

macro_rules! direction_map {
    {
        Up => $ue:expr,
        Down => $de:expr,
        East => $ee:expr,
        West => $we:expr,
        North => $ne:expr,
        South => $se:expr,
    } => {
        DirectionMap {
            values: [
                HashSet::from_iter($ue),
                HashSet::from_iter($de),
                HashSet::from_iter($ee),
                HashSet::from_iter($we),
                HashSet::from_iter($ne),
                HashSet::from_iter($se)
            ]
        }
    }
}

pub struct WfcPiece {
    allowed_neighbors: DirectionMap<HashSet<PieceId>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PieceId(usize);

pub struct WfcStep {
    piece_inventory: HashMap<PieceId, WfcPieceCount>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WfcPieceCount {
    Finite { count: usize },
    Infinite,
}

impl WfcPieceCount {
    const NONE: Self = WfcPieceCount::Finite { count: 0 };

    fn decrement(&mut self) {
        match self {
            WfcPieceCount::Finite { count } => {
                *count -= 1;
            }

            WfcPieceCount::Infinite => (),
        }
    }
}

pub struct V5 {
    nodes: Vec<Node>,
    /// Indicates the nodes that the algorithm has yet to process.
    /// They will be processed from front to back, adding new items to the back.
    queue: VecDeque<usize>,

    pieces: Vec<WfcPiece>,

    current_step: WfcStep,
    mutable_places: HashSet<IVec3>,

    step_queue: VecDeque<WfcStep>,

    pub map: TileMap3D<Option<PieceId>>,
}

impl V5 {
    pub fn new(nodes: Vec<Node>) -> Self {
        let mut queue = VecDeque::new();
        queue.push_back(0);

        const WALL: PieceId = PieceId(0);
        const FLOOR: PieceId = PieceId(1);
        const EMPTY: PieceId = PieceId(2);
        const CEILING: PieceId = PieceId(3);
        Self {
            nodes,
            queue,
            current_step: WfcStep {
                piece_inventory: HashMap::from_iter([(FLOOR, WfcPieceCount::Finite { count: 20 })]),
            },
            mutable_places: HashSet::from_iter([IVec3::ZERO]),
            step_queue: VecDeque::from_iter([WfcStep {
                piece_inventory: HashMap::from_iter([
                    (WALL, WfcPieceCount::Infinite),
                    (CEILING, WfcPieceCount::Infinite),
                ]),
            }]),
            pieces: vec![
                // wall
                WfcPiece {
                    allowed_neighbors: direction_map! {
                        Up => [WALL, CEILING],
                        Down => [WALL, CEILING],
                        East => [WALL, EMPTY],
                        West => [WALL, EMPTY],
                        North => [WALL, EMPTY],
                        South => [WALL, EMPTY],
                    },
                },
                // floor
                WfcPiece {
                    allowed_neighbors: direction_map! {
                        Up => [EMPTY],
                        Down => [CEILING],
                        East => [FLOOR, WALL],
                        West => [FLOOR, WALL],
                        North => [FLOOR, WALL],
                        South => [FLOOR, WALL],
                    },
                },
                // empty
                WfcPiece {
                    allowed_neighbors: direction_map! {
                        Up => [EMPTY],
                        Down => [EMPTY],
                        East => [EMPTY],
                        West => [EMPTY],
                        North => [EMPTY],
                        South => [EMPTY],
                    },
                },
                // ceiling
                WfcPiece {
                    allowed_neighbors: direction_map! {
                        Up => [FLOOR],
                        Down => [WALL, EMPTY],
                        East => [CEILING],
                        West => [CEILING],
                        North => [CEILING],
                        South => [CEILING],
                    },
                },
            ],
            map: Default::default(),
        }
    }

    pub fn step(&mut self) {
        let WfcStep {
            piece_inventory: ref mut inventory,
        } = self.current_step;

        for (place_idx, &place) in self.mutable_places.iter().enumerate() {
            let directions = [
                Direction::Up,
                Direction::Down,
                Direction::East,
                Direction::West,
                Direction::North,
                Direction::South,
            ];
            let mut allowed_here =
                HashSet::from_iter(inventory.iter().map(|(piece, _count)| *piece));
            for direction in directions.into_iter() {
                allowed_here = allowed_here
                    .intersection(
                        self.map
                            .cell(place + IVec3::from(direction))
                            .copied()
                            .flatten()
                            .map(|cell| self.pieces[cell.0].allowed_neighbors.clone())
                            .unwrap_or_else(|| {
                                DirectionMap::new_with_all(HashSet::from_iter(
                                    (0..self.pieces.len()).map(PieceId),
                                ))
                            })
                            .get(direction.inverse()),
                    )
                    .copied()
                    .collect();
            }
            if let Some(&piece) = allowed_here.iter().next() {
                let piece_count = inventory.get_mut(&piece).unwrap();
                piece_count.decrement();
                if *piece_count == WfcPieceCount::NONE {
                    inventory.remove(&piece);
                }
                self.map.set_cell(place, Some(piece));
            }
        }
    }
}
