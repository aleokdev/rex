use crate::math::floordiv;

use super::BigArray;
use ahash::HashMap;
use glam::{ivec3, uvec3, IVec3, UVec3};
use serde::{Deserialize, Serialize};

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

pub struct TileMap3D<Cell: Default + Copy> {
    chunks: HashMap<IVec3, TileMap3DChunk<Cell>>,
}

// For some reason cannot derive automatically
impl<Cell: Default + Copy> Default for TileMap3D<Cell> {
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
