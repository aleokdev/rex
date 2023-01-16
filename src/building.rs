use ahash::{AHashMap, AHashSet};
use glam::IVec2;
use serde::{Deserialize, Serialize};

use crate::grid::{CartesianRoomGrid, RoomId};

#[derive(Default, Serialize, Deserialize)]
pub struct BuildingMap {
    floor_maps: AHashMap<i32, FloorMap>,
}

#[derive(Serialize, Deserialize)]
pub struct FloorMap {
    grid: CartesianRoomGrid,
    room_cell_positions: Vec<AHashSet<IVec2>>,
}

impl BuildingMap {
    pub fn floor(&self, floor: i32) -> Option<&FloorMap> {
        self.floor_maps.get(&floor)
    }
    pub fn floor_mut(&mut self, floor: i32) -> Option<&mut FloorMap> {
        self.floor_maps.get_mut(&floor)
    }
    pub fn floor_entry<'s>(
        &'s mut self,
        floor: i32,
    ) -> std::collections::hash_map::Entry<'s, i32, FloorMap> {
        self.floor_maps.entry(floor)
    }
    pub fn floors(&self) -> impl ExactSizeIterator<Item = (&i32, &FloorMap)> {
        self.floor_maps.iter()
    }
}

impl FloorMap {
    pub fn new(room_count: usize) -> Self {
        Self {
            grid: Default::default(),
            room_cell_positions: vec![Default::default(); room_count],
        }
    }

    pub fn cell(&self, pos: impl Into<mint::Vector2<i32>>) -> Option<RoomId> {
        self.grid.cell(pos).copied().flatten()
    }

    pub fn cell_at(
        &self,
        chunk_pos: impl Into<mint::Vector2<i32>>,
        local_pos: impl Into<mint::Vector2<u32>>,
    ) -> Option<RoomId> {
        self.grid.cell_at(chunk_pos, local_pos).copied().flatten()
    }

    pub fn assign_cell(&mut self, pos: IVec2, value: RoomId) {
        self.room_cell_positions[value].insert(pos);
        self.grid.set_cell(pos, Some(value));
    }

    pub fn grid(&self) -> &CartesianRoomGrid {
        &self.grid
    }

    pub fn room_cell_positions(&self) -> &[AHashSet<IVec2>] {
        self.room_cell_positions.as_ref()
    }
}
