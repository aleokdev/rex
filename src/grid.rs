use ahash::AHashMap;
use glam::{ivec2, uvec2, vec2, IVec2, UVec2, Vec2};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use crate::math::floordiv;

pub type RoomId = usize;

#[derive(Serialize, Deserialize)]
pub struct GridChunk<Cell> {
    #[serde(with = "BigArray")]
    #[serde(bound = "Cell: Serialize + for<'ds> Deserialize<'ds>")]
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

    pub fn cell(&self, pos: impl Into<mint::Vector2<u32>>) -> Option<&Cell> {
        let mint::Vector2 { x, y } = pos.into();
        self.contains(mint::Vector2 { x, y })
            .then(|| self.cells.get((x + y * Self::CELLS_PER_AXIS) as usize))
            .flatten()
    }

    pub fn set_cell(&mut self, pos: impl Into<mint::Vector2<u32>>, value: Cell) {
        let mint::Vector2 { x, y } = pos.into();
        assert!(
            self.contains(mint::Vector2 { x, y }),
            "Tried to set node of wire grid which wasn't contained within"
        );
        self.cells[(x + y * Self::CELLS_PER_AXIS) as usize] = value;
    }

    pub fn contains(&self, pos: impl Into<mint::Vector2<u32>>) -> bool {
        let mint::Vector2 { x, y } = pos.into();
        x < Self::CELLS_PER_AXIS && y < Self::CELLS_PER_AXIS
    }

    pub fn cells(&self) -> impl Iterator<Item = (UVec2, &Cell)> {
        self.cells.iter().enumerate().map(|(i, cell)| {
            (
                uvec2(
                    i as u32 % Self::CELLS_PER_AXIS,
                    i as u32 / Self::CELLS_PER_AXIS,
                ),
                cell,
            )
        })
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct CartesianGrid<Cell: Default + Copy> {
    #[serde(bound = "Cell: Serialize + for<'ds> Deserialize<'ds>")]
    #[serde(with = "serialize")]
    pub chunks: AHashMap<IVec2, GridChunk<Cell>>,
}

pub mod serialize {
    use std::marker::PhantomData;

    use ahash::AHashMap;
    use glam::{ivec2, IVec2};
    use serde::{de::Visitor, ser::SerializeMap, Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer, T: Serialize>(
        t: &AHashMap<IVec2, T>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(t.len()))?;
        for (k, v) in t {
            map.serialize_entry(&format!("{},{}", k.x, k.y), v)?;
        }
        map.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
        d: D,
    ) -> Result<AHashMap<IVec2, T>, D::Error> {
        struct V<T> {
            marker: PhantomData<fn() -> T>,
        }
        impl<'de, T: Deserialize<'de>> Visitor<'de> for V<T> {
            type Value = AHashMap<IVec2, T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a key-value map with stringified vectors as key")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut res = AHashMap::with_capacity(map.size_hint().unwrap_or(0));
                while let Some((key, value)) = map.next_entry::<String, T>()? {
                    let (x, y) = key.split_once(',').unwrap();
                    let (x, y) = (x.parse().unwrap(), y.parse().unwrap());
                    res.insert(ivec2(x, y), value);
                }

                Ok(res)
            }
        }
        d.deserialize_map(V {
            marker: PhantomData::default(),
        })
    }
}

impl<Cell: Default + Copy> CartesianGrid<Cell> {
    pub fn cell(&self, pos: impl Into<mint::Vector2<i32>>) -> Option<&Cell> {
        let (chunk_pos, local_pos) = Self::global_coords_to_chunk_and_local_coords(pos.into());

        self.cell_at(chunk_pos, local_pos)
    }

    pub fn cell_at(
        &self,
        chunk_pos: impl Into<mint::Vector2<i32>>,
        local_pos: impl Into<mint::Vector2<u32>>,
    ) -> Option<&Cell> {
        self.chunks.get(&chunk_pos.into().into()).map(|chunk| {
            chunk
                .cell(local_pos)
                .expect("Cell calculations aren't correct")
        })
    }

    pub fn set_cell(&mut self, pos: impl Into<mint::Vector2<i32>>, value: Cell) {
        let (chunk_pos, local_pos) = Self::global_coords_to_chunk_and_local_coords(pos.into());

        self.set_cell_at(chunk_pos, local_pos, value)
    }

    pub fn set_cell_at(
        &mut self,
        chunk_pos: impl Into<mint::Vector2<i32>>,
        local_pos: impl Into<mint::Vector2<u32>>,
        value: Cell,
    ) {
        self.chunks
            .entry(chunk_pos.into().into())
            .or_insert_with(|| Default::default())
            .set_cell(local_pos, value)
    }

    pub fn global_coords_to_chunk_and_local_coords(
        vec: impl Into<mint::Vector2<i32>>,
    ) -> (IVec2, UVec2) {
        let mint::Vector2 { x, y } = vec.into();
        let x_c = floordiv(x, GridChunk::<Cell>::CELLS_PER_AXIS as i32);
        let x_local = x - x_c * GridChunk::<Cell>::CELLS_PER_AXIS as i32;
        let y_c = floordiv(y, GridChunk::<Cell>::CELLS_PER_AXIS as i32);
        let y_local = y - y_c * GridChunk::<Cell>::CELLS_PER_AXIS as i32;
        assert!(x_local >= 0 && x_local < GridChunk::<Cell>::CELLS_PER_AXIS as i32);
        assert!(y_local >= 0 && y_local < GridChunk::<Cell>::CELLS_PER_AXIS as i32);
        (ivec2(x_c, y_c), uvec2(x_local as u32, y_local as u32))
    }

    pub fn chunk_and_local_coords_to_global_coords(
        chunk_pos: impl Into<mint::Vector2<i32>>,
        local_pos: impl Into<mint::Vector2<u32>>,
    ) -> mint::Vector2<i32> {
        (<mint::Vector2<i32> as Into<IVec2>>::into(chunk_pos.into())
            * GridChunk::<Cell>::CELLS_PER_AXIS as i32
            + <mint::Vector2<u32> as Into<UVec2>>::into(local_pos.into()).as_ivec2())
        .into()
    }

    pub fn cells(&self) -> impl Iterator<Item = (IVec2, &Cell)> {
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
    outline: Vec<Vec2>,
}

impl RoomTemplate {
    pub fn with_inscribed_radius_and_outline<T: Into<mint::Vector2<f32>> + Copy>(
        grid: CartesianGrid<RoomTemplateCell>,
        inscribed_radius: f32,
        outline: &[T],
    ) -> Self {
        Self {
            grid,
            inscribed_radius,
            outline: outline.into_iter().map(|&x| x.into().into()).collect(),
        }
    }

    pub fn inscribed_radius(&self) -> f32 {
        self.inscribed_radius
    }

    pub fn grid(&self) -> &CartesianGrid<RoomTemplateCell> {
        &self.grid
    }

    pub fn outline(&self) -> &[Vec2] {
        self.outline.as_ref()
    }
}

/// The template didn't fit at the position given since part of the space it needed was already occupied.
#[derive(Clone, Copy, Debug)]
pub struct PlaceTemplateError;

pub type CartesianRoomGrid = CartesianGrid<Option<RoomId>>;

impl CartesianRoomGrid {
    pub fn place_template_near(
        &mut self,
        template: &RoomTemplate,
        id: RoomId,
        position: impl Into<mint::Vector2<f32>>,
        rng: &mut impl rand::Rng,
    ) -> IVec2 {
        let position = Vec2::from(position.into());
        let random_angle = rng.gen_range(0.0..(std::f32::consts::PI * 2.));
        let random_dir = vec2(random_angle.cos(), random_angle.sin());
        const STEP: f32 = 1.;

        let mut try_pos = position + random_dir * template.inscribed_radius();
        while let Err(_) = self.place_template_at(template, id, try_pos.as_ivec2()) {
            try_pos += STEP * random_dir;
        }

        try_pos.as_ivec2()
    }

    pub fn place_template_at(
        &mut self,
        template: &RoomTemplate,
        id: RoomId,
        template_position: impl Into<mint::Vector2<i32>>,
    ) -> Result<(), PlaceTemplateError> {
        let template_position = IVec2::from(template_position.into());
        // We allocate a vector of all the positions we change just in case we need to rollback.
        let mut positions_changed = Vec::with_capacity(
            template.grid().chunks.len() * GridChunk::<RoomTemplateCell>::CELL_COUNT as usize,
        );

        for (&chunk_pos, template_chunk) in template.grid().chunks.iter() {
            for x in 0..GridChunk::<RoomId>::CELLS_PER_AXIS {
                for y in 0..GridChunk::<RoomId>::CELLS_PER_AXIS {
                    let local_pos = uvec2(x, y);
                    if template_chunk.cell(local_pos).unwrap().is_occupied() {
                        let global_pos = IVec2::from(
                            Self::chunk_and_local_coords_to_global_coords(chunk_pos, local_pos),
                        ) + template_position;
                        let cell = self.cell(global_pos).copied().flatten();
                        if cell.is_some() {
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

        Ok(())
    }
}
