use std::collections::HashMap;

use glam::{ivec2, uvec2, vec2, IVec2, UVec2, Vec2};

pub type RoomId = usize;

struct GridChunk<Cell> {
    cells: [Cell; 256], // TODO: Use Self::CELL_COUNT when Rust stabilizes using generic Self types in anonymous constants
}

impl<Cell: Default + Copy> Default for GridChunk<Cell> {
    fn default() -> Self {
        Self {
            cells: [Default::default(); 256],
        }
    }
}

impl<Cell> GridChunk<Cell> {
    /// Size of the chunk in cells
    pub const CELLS_PER_AXIS: u32 = 16;
    /// Total cells in the chunk
    pub const CELL_COUNT: u32 = Self::CELLS_PER_AXIS * Self::CELLS_PER_AXIS;

    pub fn cell(&self, pos @ UVec2 { x, y }: UVec2) -> Option<&Cell> {
        self.contains(pos)
            .then(|| self.cells.get((x + y * Self::CELLS_PER_AXIS) as usize))
            .flatten()
    }

    pub fn set_cell(&mut self, pos @ UVec2 { x, y }: UVec2, value: Cell) {
        assert!(
            self.contains(pos),
            "Tried to set node of wire grid which wasn't contained within"
        );
        self.cells[(x + y * Self::CELLS_PER_AXIS) as usize] = value;
    }

    pub fn contains(&self, UVec2 { x, y }: UVec2) -> bool {
        x < Self::CELLS_PER_AXIS && y < Self::CELLS_PER_AXIS
    }
}

#[derive(Default)]
pub struct CartesianGrid<Cell: Default + Copy> {
    chunks: HashMap<IVec2, GridChunk<Cell>>,
}

impl<Cell: Default + Copy> CartesianGrid<Cell> {
    pub fn cell(&self, pos: IVec2) -> Option<&Cell> {
        let (chunk_pos, local_pos) = Self::global_coords_to_chunk_and_local_coords(pos);

        self.cell_at(chunk_pos, local_pos)
    }

    pub fn cell_at(&self, chunk_pos: IVec2, local_pos: UVec2) -> Option<&Cell> {
        self.chunks.get(&chunk_pos).map(|chunk| {
            chunk
                .cell(local_pos)
                .expect("Cell calculations aren't correct")
        })
    }

    pub fn set_cell(&mut self, pos: IVec2, value: Cell) {
        let (chunk_pos, local_pos) = Self::global_coords_to_chunk_and_local_coords(pos);

        self.set_cell_at(chunk_pos, local_pos, value)
    }

    pub fn set_cell_at(&mut self, chunk_pos: IVec2, local_pos: UVec2, value: Cell) {
        self.chunks
            .entry(chunk_pos)
            .or_insert_with(|| Default::default())
            .set_cell(local_pos, value)
    }

    pub fn global_coords_to_chunk_and_local_coords(IVec2 { x, y }: IVec2) -> (IVec2, UVec2) {
        let x_c = floordiv(x, GridChunk::<Cell>::CELLS_PER_AXIS as i32);
        let x_local = x - x_c * GridChunk::<Cell>::CELLS_PER_AXIS as i32;
        let y_c = floordiv(y, GridChunk::<Cell>::CELLS_PER_AXIS as i32);
        let y_local = y - y_c * GridChunk::<Cell>::CELLS_PER_AXIS as i32;
        (ivec2(x_c, y_c), uvec2(x_local as u32, y_local as u32))
    }

    pub fn chunk_and_local_coords_to_global_coords(chunk_pos: IVec2, local_pos: UVec2) -> IVec2 {
        chunk_pos * GridChunk::<Cell>::CELLS_PER_AXIS as i32 + local_pos.as_ivec2()
    }
}

fn floordiv(a: i32, b: i32) -> i32 {
    (a - if a < 0 { b - 1 } else { 0 }) / b
}

#[derive(Default, Clone, Copy)]
pub enum RoomTemplateCell {
    Occupied,
    #[default]
    Vacant,
}

impl RoomTemplateCell {
    pub fn is_occupied(&self) -> bool {
        match self {
            RoomTemplateCell::Occupied => true,
            RoomTemplateCell::Vacant => false,
        }
    }

    pub fn is_vacant(&self) -> bool {
        match self {
            RoomTemplateCell::Occupied => false,
            RoomTemplateCell::Vacant => true,
        }
    }
}

pub struct RoomTemplate {
    grid: CartesianGrid<RoomTemplateCell>,
    /// Given the template's grid, the radius of the biggest circle that can fit within all cells marked with `true` from the origin.
    inscribed_radius: f32,
}

impl RoomTemplate {
    pub fn with_inscribed_radius(
        grid: CartesianGrid<RoomTemplateCell>,
        inscribed_radius: f32,
    ) -> Self {
        Self {
            grid,
            inscribed_radius,
        }
    }

    pub fn inscribed_radius(&self) -> f32 {
        self.inscribed_radius
    }

    pub fn grid(&self) -> &CartesianGrid<RoomTemplateCell> {
        &self.grid
    }
}

/// The template didn't fit at the position given since part of the space it needed was already occupied.
pub struct PlaceTemplateError;

pub type CartesianRoomGrid = CartesianGrid<Option<RoomId>>;

impl CartesianRoomGrid {
    pub fn place_template_near(
        &mut self,
        template: &RoomTemplate,
        id: RoomId,
        position: Vec2,
        rng: &mut impl rand::Rng,
    ) {
        let random_angle = rng.gen_range(0.0..(std::f32::consts::PI * 2.));
        let random_dir = vec2(random_angle.cos(), random_angle.sin());
        const STEP: f32 = 1.;

        let mut try_pos = position + random_dir * template.inscribed_radius();
        while let Err(_) = self.place_template_at(template, id, try_pos.as_ivec2()) {
            try_pos += STEP * random_dir;
        }
    }

    pub fn place_template_at(
        &mut self,
        template: &RoomTemplate,
        id: RoomId,
        template_position: IVec2,
    ) -> Result<(), PlaceTemplateError> {
        // We allocate a vector of all the positions we change just in case we need to rollback.
        let mut positions_changed = Vec::with_capacity(
            template.grid().chunks.len() * GridChunk::<RoomTemplateCell>::CELL_COUNT as usize,
        );

        for (&chunk_pos, template_chunk) in template.grid().chunks.iter() {
            for x in 0..GridChunk::<RoomId>::CELLS_PER_AXIS {
                for y in 0..GridChunk::<RoomId>::CELLS_PER_AXIS {
                    let local_pos = uvec2(x, y);
                    if template_chunk.cell(local_pos).unwrap().is_occupied() {
                        let global_pos =
                            Self::chunk_and_local_coords_to_global_coords(chunk_pos, local_pos)
                                + template_position;
                        if self.cell(global_pos).is_some() {
                            for global_pos in positions_changed {
                                self.set_cell(global_pos, None)
                            }
                            return Err(PlaceTemplateError);
                        } else {
                            self.set_cell(global_pos, Some(id));
                            positions_changed.push(global_pos);
                        }
                    }
                }
            }
        }

        todo!()
    }
}
