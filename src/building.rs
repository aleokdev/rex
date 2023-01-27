use ahash::{AHashMap, AHashSet};
use glam::{IVec2, IVec3};
use serde::{Deserialize, Serialize};

use crate::{
    grid::{CartesianRoomGrid, RoomId},
    node::NodeId,
};

pub type FloorIdx = i32;

/// Describes an entire building's floor plan.
///
/// This structure holds all grid data, and determines where rooms go.
#[derive(Default, Serialize, Deserialize)]
pub struct BuildingMap {
    floor_maps: AHashMap<FloorIdx, FloorMap>,
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

/// Describes a floor plan.
#[derive(Serialize, Deserialize, Default)]
pub struct FloorMap {
    grid: CartesianRoomGrid,
    room_cell_positions: AHashMap<RoomId, AHashSet<IVec2>>,
}

impl FloorMap {
    pub fn new() -> Self {
        Self::default()
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
        self.room_cell_positions
            .entry(value)
            .or_default()
            .insert(pos);
        self.grid.set_cell(pos, Some(value));
    }

    pub fn grid(&self) -> &CartesianRoomGrid {
        &self.grid
    }

    pub fn room_cell_positions(&self) -> &AHashMap<RoomId, AHashSet<IVec2>> {
        &self.room_cell_positions
    }
}

/// A room inside a building, related to a specific node.
///
/// It holds room-specific data, such as door (& teleporter: TODO) positions.
// TODO: Store teleporter positions
#[derive(Clone, Serialize, Deserialize)]
pub struct Room {
    /// Door positions, in door space.
    pub door_positions: AHashSet<IVec3>,
    node: NodeId,
    /// The position this room started expanding from.
    ///
    /// It is guaranteed that this position will contain a cell with the ID of this room.
    starting_pos: IVec3,
}

impl Room {
    pub fn new(node: NodeId, starting_pos: IVec3) -> Self {
        Self {
            door_positions: Default::default(),
            node,
            starting_pos,
        }
    }

    pub fn starting_pos(&self) -> IVec3 {
        self.starting_pos
    }

    pub fn node(&self) -> NodeId {
        self.node
    }
}
