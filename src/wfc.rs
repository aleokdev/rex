use ahash::{HashMap, HashSet};
use enum_map::Enum;
use glam::IVec3;
use rand::{seq::IteratorRandom, Rng};

#[derive(Clone)]
pub struct Tile {
    adjacency_rules: SpatialRuleMap,
}

#[derive(Clone)]
pub struct SpatialRuleMap {
    /// Key represents offset from here, and value represents the tiles that can appear in that space.
    rules: HashMap<IVec3, HashSet<TileId>>,
}

impl SpatialRuleMap {
    pub fn from_default(default_rule: &HashSet<TileId>) -> Self {
        Self::from_fn(|_| default_rule.clone())
    }

    pub fn from_fn(mut f: impl FnMut(IVec3) -> HashSet<TileId>) -> Self {
        let mut result =
            HashMap::with_capacity_and_hasher(3 * 3 * 3 - 1, ahash::RandomState::new());
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    if x == 0 && y == 0 && z == 0 {
                        continue;
                    }

                    let pos = glam::ivec3(x, y, z);
                    result.insert(pos, f(pos));
                }
            }
        }

        Self { rules: result }
    }

    pub fn restricted(mut self, offset: IVec3, rule: HashSet<TileId>) -> Self {
        self.rules.insert(offset, rule);
        self
    }

    /// Rotates clockwise around the Y axis.
    pub fn rotated_clockwise(&self) -> Self {
        SpatialRuleMap::from_fn(|pos| {
            // New +X is old -Z
            // New +Z is old +X
            let rotated = glam::ivec3(-pos.z, pos.y, pos.x);
            self.rules[&rotated].clone()
        })
    }
}

impl Tile {
    pub fn from_rules(adjacency_rules: SpatialRuleMap) -> Self {
        Self { adjacency_rules }
    }
}

pub type TileId = usize;

#[derive(Clone, Copy, Enum)]
pub enum Direction {
    Up,
    Down,

    Forward,
    Backward,

    Right,
    Left,
}

impl From<Direction> for IVec3 {
    fn from(dir: Direction) -> glam::IVec3 {
        match dir {
            // TODO: Change these to be actually correct
            Direction::Up => IVec3::Y,
            Direction::Down => IVec3::NEG_Y,
            Direction::Forward => IVec3::Z,
            Direction::Backward => IVec3::NEG_Z,
            Direction::Right => IVec3::X,
            Direction::Left => IVec3::NEG_X,
        }
    }
}

#[derive(Default, Clone)]
pub struct Context {
    tiles: Vec<Tile>,
}

impl Context {
    pub fn new(tiles: Vec<Tile>) -> Self {
        Self { tiles }
    }
}

#[derive(Debug)]
pub enum SpaceTile {
    Collapsed(TileId),
    NonCollapsed(HashSet<TileId>),
}

#[derive(Default, Debug)]
pub struct World {
    pub space: HashMap<IVec3, SpaceTile>,
}

#[derive(thiserror::Error, Debug)]
#[error("rule contradiction")]
pub struct Error;

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns an iterator of the non-collapsed space with least tile options to choose from.
    ///
    /// If there are no tiles placed yet or all tiles are collapsed, this will return [`None`].
    pub fn least_entropy_positions(&self) -> Option<impl Iterator<Item = IVec3> + '_> {
        let min_entropy = self
            .space
            .iter()
            .filter_map(|(_pos, tile)| {
                if let SpaceTile::NonCollapsed(h) = tile {
                    (!h.is_empty()).then_some(h)
                } else {
                    None
                }
            })
            .map(|h| h.len())
            .min()?;

        Some(self.space.iter().filter_map(move |(pos, tile)| {
            if let SpaceTile::NonCollapsed(h) = tile {
                (h.len() == min_entropy).then_some(*pos)
            } else {
                None
            }
        }))
    }

    /// Returns an iterator of the non-collapsed space with least tile options to choose from on the Y=0 plane.
    ///
    /// If there are no tiles placed yet or all tiles are collapsed, this will return [`None`].
    pub fn least_entropy_positions_2d(&self) -> Option<impl Iterator<Item = IVec3> + '_> {
        let min_entropy = self
            .space
            .iter()
            .filter(|(pos, _)| pos.y == 0)
            .filter_map(|(_pos, tile)| {
                if let SpaceTile::NonCollapsed(h) = tile {
                    (!h.is_empty()).then_some(h)
                } else {
                    None
                }
            })
            .map(|h| h.len())
            .min()?;

        Some(
            self.space
                .iter()
                .filter(|(pos, _)| pos.y == 0)
                .filter_map(move |(pos, tile)| {
                    if let SpaceTile::NonCollapsed(h) = tile {
                        (h.len() == min_entropy).then_some(*pos)
                    } else {
                        None
                    }
                }),
        )
    }

    pub fn collapse(
        &mut self,
        ctx: &Context,
        rng: &mut impl Rng,
        position: IVec3,
    ) -> Result<TileId, Error> {
        let tile = self.get_or_insert_at(ctx, position);
        match tile {
            SpaceTile::Collapsed(entry) => Ok(*entry),
            SpaceTile::NonCollapsed(possibilities) => {
                let result = possibilities.iter().choose(rng).copied().ok_or(Error)?;
                *tile = SpaceTile::Collapsed(result);
                self.propagate_rules(ctx, position, result);
                Ok(result)
            }
        }
    }

    pub fn place(&mut self, ctx: &Context, tile_id: TileId, position: IVec3) -> Result<(), Error> {
        let tile = self.get_or_insert_at(ctx, position);

        match tile {
            SpaceTile::Collapsed(_) => return Err(Error),
            SpaceTile::NonCollapsed(possibilities) if !possibilities.contains(&tile_id) => {
                return Err(Error)
            }
            _ => (),
        }
        *tile = SpaceTile::Collapsed(tile_id);
        self.propagate_rules(ctx, position, tile_id);

        Ok(())
    }

    fn propagate_rules(&mut self, ctx: &Context, position: IVec3, tile_id: TileId) {
        let rulemap = &ctx.tiles[tile_id].adjacency_rules;
        for (&offset, allowed) in rulemap.rules.iter() {
            let rule_pos = position + offset;
            match self.get_or_insert_at(ctx, rule_pos) {
                SpaceTile::Collapsed(_) => (),
                SpaceTile::NonCollapsed(adj_rules) => {
                    // Only allow neighbour tiles stated in the rule map
                    *adj_rules = adj_rules
                        .intersection(allowed)
                        .copied()
                        // Neighbour tiles also have their own rules
                        // We assume those are respected by the rules of this tile
                        /*
                        .filter(|&tile| {
                            ctx.tiles[tile].adjacency_rules.rules[&-offset].contains(&tile_id)
                        })*/
                        .collect();
                }
            }
        }
    }

    fn get_or_insert_at(&mut self, ctx: &Context, position: glam::IVec3) -> &mut SpaceTile {
        if !self.space.contains_key(&position) {
            let tiles_allowed_here: HashSet<TileId> = (0..ctx.tiles.len())
                .filter(|&id| self.do_rules_apply(position, &ctx.tiles[id].adjacency_rules))
                .collect();
            self.space
                .insert(position, SpaceTile::NonCollapsed(tiles_allowed_here));
        }
        self.space.get_mut(&position).unwrap()
    }

    fn do_rules_apply(&self, position: IVec3, rulemap: &SpatialRuleMap) -> bool {
        for (&offset, allowed) in rulemap.rules.iter() {
            let rule_pos = position + offset;
            match self.space.get(&rule_pos) {
                Some(SpaceTile::Collapsed(tile)) => {
                    if !allowed.contains(tile) {
                        return false;
                    }
                }
                _ => (),
            }
        }

        true
    }
}
