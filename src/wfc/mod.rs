pub mod tilemap;

use std::{collections::VecDeque, default, num::NonZeroUsize, ops::Not, path::PathBuf};

use ahash::{HashMap, HashSet};
use glam::{ivec3, uvec2, uvec3, IVec3, UVec2, UVec3, Vec2};
use rand::seq::{IteratorRandom, SliceRandom};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use crate::{math::floordiv, node::Node};

use self::tilemap::TileMap3D;

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
                PieceIdSet::from_iter($ue),
                PieceIdSet::from_iter($de),
                PieceIdSet::from_iter($ee),
                PieceIdSet::from_iter($we),
                PieceIdSet::from_iter($ne),
                PieceIdSet::from_iter($se)
            ]
        }
    }
}

pub struct WfcPiece {
    allowed_neighbors: DirectionMap<PieceIdSet>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PieceId(pub usize);

#[derive(Clone, Copy)]
pub struct PieceIdSet {
    possibilities: [bool; 64],
}

impl PieceIdSet {
    pub fn with_everything_allowed() -> Self {
        Self {
            possibilities: [true; 64],
        }
    }
    pub fn with_nothing_allowed() -> Self {
        Self {
            possibilities: [false; 64],
        }
    }

    #[must_use]
    pub fn collapse(&self, rng: &mut impl rand::Rng, piece_count: usize) -> PieceId {
        let allowed = self.possibilities[0..piece_count]
            .iter()
            .enumerate()
            .filter_map(|(idx, allowed)| allowed.then_some(PieceId(idx)));
        allowed
            .choose(rng)
            .expect("uncollapsed piece id had no possibilities")
    }

    #[must_use]
    pub fn collapse_first(&self, piece_count: usize) -> PieceId {
        self.possibilities[0..piece_count]
            .iter()
            .enumerate()
            .find_map(|(idx, allowed)| allowed.then_some(PieceId(idx)))
            .expect("uncollapsed piece id had no possibilities")
    }

    #[must_use]
    pub fn without(&self, id: PieceId) -> Self {
        let mut ret = *self;
        ret.possibilities[id.0] = false;
        ret
    }

    #[must_use]
    pub fn with(&self, id: PieceId) -> Self {
        let mut ret = *self;
        ret.possibilities[id.0] = true;
        ret
    }

    #[must_use]
    pub fn inverse(&self) -> Self {
        let mut ret = *self;
        for p in ret.possibilities.iter_mut() {
            *p = !*p;
        }
        ret
    }

    #[must_use]
    pub fn intersect(&self, other: PieceIdSet) -> Self {
        let mut ret = *self;
        for (p, o) in ret
            .possibilities
            .iter_mut()
            .zip(other.possibilities.into_iter())
        {
            *p = *p && o;
        }
        ret
    }

    #[must_use]
    pub fn filter_allowed(&self, piece_count: usize, f: impl Fn(PieceId) -> bool) -> Self {
        let mut ret = *self;
        for (id, p) in ret.possibilities[0..piece_count].iter_mut().enumerate() {
            if *p {
                *p = *p && f(PieceId(id));
            }
        }
        ret
    }

    #[must_use]
    pub fn contains(&self, id: PieceId) -> bool {
        self.possibilities[id.0]
    }

    #[must_use]
    pub fn count_allowed(&self, piece_count: usize) -> usize {
        self.possibilities[0..piece_count]
            .iter()
            .filter(|&&x| x)
            .count()
    }

    #[must_use]
    pub fn is_empty(&self, piece_count: usize) -> bool {
        self.count_allowed(piece_count) == 0
    }

    #[must_use]
    pub fn iter_allowed<'a>(&'a self, piece_count: usize) -> impl Iterator<Item = PieceId> + 'a {
        self.possibilities[0..piece_count]
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(id, allowed)| allowed.then_some(PieceId(id)))
    }
}

impl FromIterator<PieceId> for PieceIdSet {
    fn from_iter<T: IntoIterator<Item = PieceId>>(iter: T) -> Self {
        iter.into_iter()
            .fold(PieceIdSet::with_nothing_allowed(), |p, id| p.with(id))
    }
}

#[derive(Clone, Copy, Default)]
pub enum WfcCell {
    #[default]
    Undiscovered,
    Uncollapsed(PieceIdSet),
    Collapsed(PieceId),
}

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
    uncollapsed_places: HashSet<IVec3>,

    step_queue: VecDeque<WfcStep>,
    history: Vec<PlacedPiece>,

    pub map: TileMap3D<WfcCell>,
}

pub struct PlacedPiece {
    position: IVec3,
}

impl V5 {
    pub fn new(nodes: Vec<Node>) -> Self {
        let mut queue = VecDeque::new();
        queue.push_back(0);

        let mut map = TileMap3D::default();
        map.set_cell(
            IVec3::ZERO,
            WfcCell::Uncollapsed(PieceIdSet::from_iter([WALL, FLOOR])),
        );

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
            uncollapsed_places: HashSet::from_iter([IVec3::ZERO]),
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
                        East => [WALL, EMPTY, FLOOR],
                        West => [WALL, EMPTY, FLOOR],
                        North => [WALL, EMPTY, FLOOR],
                        South => [WALL, EMPTY, FLOOR],
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
            history: Default::default(),
            map,
        }
    }

    pub fn step(&mut self) {
        let mut rng = rand::thread_rng();

        if let Some(&place) = self.uncollapsed_places.iter().choose(&mut rng) {
            let cell = self.map.cell(place);
            let cell = match cell {
                Some(WfcCell::Uncollapsed(uncollapsed)) => *uncollapsed,
                Some(WfcCell::Collapsed(_)) => {
                    panic!("collapsed cell was found in uncollapsed place")
                }
                None | Some(WfcCell::Undiscovered) => PieceIdSet::with_everything_allowed(),
            };
            let collapsed = cell.collapse(&mut rng, 4);
            let collapsed_piece = &self.pieces[collapsed.0];
            let directions = [
                // TODO enable 3d:
                //Direction::Up,
                //Direction::Down,
                Direction::East,
                Direction::West,
                Direction::North,
                Direction::South,
            ];
            for direction in directions.into_iter() {
                let there = place + IVec3::from(direction);
                let allowed_there = match self.map.cell(there) {
                    Some(WfcCell::Uncollapsed(uncollapsed)) => *uncollapsed,
                    Some(WfcCell::Collapsed(_)) => continue,
                    Some(WfcCell::Undiscovered) | None => PieceIdSet::with_everything_allowed(),
                };
                let allowed_there =
                    allowed_there.intersect(*collapsed_piece.allowed_neighbors.get(direction));
                let allowed_there = allowed_there.filter_allowed(4, |id| {
                    self.pieces[id.0]
                        .allowed_neighbors
                        .get(direction.inverse())
                        .contains(collapsed)
                });

                self.uncollapsed_places.insert(there);

                self.map
                    .set_cell(there, WfcCell::Uncollapsed(allowed_there));
            }

            self.uncollapsed_places.remove(&place);
            self.map.set_cell(place, WfcCell::Collapsed(collapsed));
        }
    }
}
