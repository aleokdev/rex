//! The simplest possible example that does something.
#![allow(clippy::unnecessary_wraps)]

use std::{collections::HashSet, path::PathBuf, sync::mpsc, thread};

use ggez::{
    conf::{WindowMode, WindowSetup},
    event,
    glam::*,
    graphics::{self, DrawMode, Rect},
    input, Context,
};
use rand::seq::IteratorRandom;
use rex::wfc::{self, Tile, TileId};

struct MainState {
    pos: Vec2,
    scale: f32,
    mesh: graphics::Mesh,
    mesh_producer: Option<mpsc::Receiver<graphics::MeshBuilder>>,
    wfc: wfc::Context,
}

impl MainState {
    fn new(path: &std::path::Path, ctx: &mut Context) -> anyhow::Result<MainState> {
        fn any() -> HashSet<TileId, ahash::RandomState> {
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].into_iter().collect()
        }
        fn not(t: &HashSet<TileId, ahash::RandomState>) -> HashSet<TileId, ahash::RandomState> {
            any().difference(t).copied().collect()
        }
        fn any_east_connectable() -> HashSet<TileId, ahash::RandomState> {
            [0, 2, 4, 5, 7, 8, 11].into_iter().collect()
        }
        fn any_south_connectable() -> HashSet<TileId, ahash::RandomState> {
            [1, 2, 3, 5, 8, 9, 11].into_iter().collect()
        }
        fn any_west_connectable() -> HashSet<TileId, ahash::RandomState> {
            [0, 2, 3, 4, 6, 9, 11].into_iter().collect()
        }
        fn any_north_connectable() -> HashSet<TileId, ahash::RandomState> {
            [1, 3, 4, 5, 6, 7, 11].into_iter().collect()
        }
        fn connect_thru(north: bool, east: bool, south: bool, west: bool) -> Tile {
            let mut result = wfc::SpatialRuleMap::from_default(&any());
            result = if north {
                result.restricted(glam::IVec3::NEG_Z, any_south_connectable())
            } else {
                result.restricted(glam::IVec3::NEG_Z, not(&any_south_connectable()))
            };
            result = if east {
                result.restricted(glam::IVec3::X, any_west_connectable())
            } else {
                result.restricted(glam::IVec3::X, not(&any_west_connectable()))
            };
            result = if south {
                result.restricted(glam::IVec3::Z, any_north_connectable())
            } else {
                result.restricted(glam::IVec3::Z, not(&any_north_connectable()))
            };
            result = if west {
                result.restricted(glam::IVec3::NEG_X, any_east_connectable())
            } else {
                result.restricted(glam::IVec3::NEG_X, not(&any_east_connectable()))
            };

            Tile::from_rules(result)
        }
        let horizontal = connect_thru(false, true, false, true);
        let vertical = connect_thru(true, false, true, false);

        let s_t = connect_thru(false, true, true, true);
        let w_t = connect_thru(true, false, true, true);
        let n_t = connect_thru(true, true, false, true);
        let e_t = connect_thru(true, true, true, false);

        let nw_t = connect_thru(true, false, false, true);
        let ne_t = connect_thru(true, true, false, false);
        let se_t = connect_thru(false, true, true, false);
        let sw_t = connect_thru(false, false, true, true);

        let empty = connect_thru(false, false, false, false);
        let full = connect_thru(true, true, true, true);

        let wfc = wfc::Context::new(vec![
            horizontal, vertical, s_t, w_t, n_t, e_t, nw_t, ne_t, se_t, sw_t, empty, full,
        ]);

        let rx = spawn_mesh_builder(wfc.clone());

        Ok(MainState {
            pos: vec2(-25., -25.),
            scale: 20.,
            mesh: graphics::Mesh::from_data(
                ctx,
                graphics::MeshData {
                    vertices: &[],
                    indices: &[],
                },
            ),
            mesh_producer: Some(rx),
            wfc,
        })
    }
}

fn spawn_mesh_builder(ctx: wfc::Context) -> mpsc::Receiver<graphics::MeshBuilder> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut world = wfc::World::default();
        world.place(&ctx, 1, glam::IVec3::ZERO).unwrap();
        world.place(&ctx, 2, glam::ivec3(2, 0, 3)).unwrap();
        world.place(&ctx, 10, glam::ivec3(-2, 0, -3)).unwrap();
        world.place(&ctx, 4, glam::ivec3(-3, 0, 5)).unwrap();
        let mut rng = rand::thread_rng();

        let mut builder = graphics::MeshBuilder::new();
        build_world_mesh(&mut builder, &world).unwrap();
        tx.send(builder).unwrap();
        thread::sleep(std::time::Duration::from_millis(1000));
        let start = std::time::Instant::now();

        for iter in 0..10000 {
            let random_min_entropy_pos = match world.least_entropy_positions_2d() {
                Some(x) => x.choose(&mut rng).unwrap(),
                None => glam::IVec3::ZERO,
            };
            world
                .collapse(&ctx, &mut rng, random_min_entropy_pos)
                .unwrap();
            if iter % 100 == 0 {
                let mut builder = graphics::MeshBuilder::new();
                build_world_mesh(&mut builder, &world).unwrap();
                tx.send(builder).unwrap();
            }
        }

        let end = std::time::Instant::now();
        log::info!("Finished collapsing in {:2}s", (end - start).as_secs_f32());
    });
    rx
}

fn build_world_mesh(builder: &mut graphics::MeshBuilder, world: &wfc::World) -> anyhow::Result<()> {
    for (pos, tile) in world.space.iter() {
        if pos.y != 0 {
            continue;
        }
        match tile {
            &wfc::SpaceTile::Collapsed(index) => {
                match index {
                    0 | 2 | 3 | 4 | 6 | 9 | 11 => {
                        builder.line(
                            &[
                                [pos.x as f32, pos.z as f32 + 0.5],
                                [pos.x as f32 + 0.5, pos.z as f32 + 0.5],
                            ],
                            0.3,
                            graphics::Color::BLACK,
                        )?;
                    }
                    _ => (),
                }
                match index {
                    1 | 3 | 4 | 5 | 6 | 7 | 11 => {
                        builder.line(
                            &[
                                [pos.x as f32 + 0.5, pos.z as f32],
                                [pos.x as f32 + 0.5, pos.z as f32 + 0.5],
                            ],
                            0.3,
                            graphics::Color::BLACK,
                        )?;
                    }
                    _ => (),
                }
                match index {
                    0 | 2 | 4 | 5 | 7 | 8 | 11 => {
                        builder.line(
                            &[
                                [pos.x as f32 + 0.5, pos.z as f32 + 0.5],
                                [pos.x as f32 + 1.0, pos.z as f32 + 0.5],
                            ],
                            0.3,
                            graphics::Color::BLACK,
                        )?;
                    }
                    _ => (),
                }
                match index {
                    1 | 2 | 3 | 5 | 8 | 9 | 11 => {
                        builder.line(
                            &[
                                [pos.x as f32 + 0.5, pos.z as f32 + 0.5],
                                [pos.x as f32 + 0.5, pos.z as f32 + 1.0],
                            ],
                            0.3,
                            graphics::Color::BLACK,
                        )?;
                    }
                    _ => (),
                }
            }
            wfc::SpaceTile::NonCollapsed(h) => {
                let mut color = ggez::graphics::Color::WHITE;
                color.a = (h.len() as f32 / 3.0).min(1.0);
                builder.rectangle(
                    DrawMode::fill(),
                    Rect::new(pos.x as f32, pos.z as f32, 1., 1.),
                    color,
                )?;
            }
        };
    }

    Ok(())
}

impl event::EventHandler<anyhow::Error> for MainState {
    fn update(&mut self, ctx: &mut Context) -> anyhow::Result<()> {
        if ctx.mouse.button_pressed(input::mouse::MouseButton::Middle) {
            let x: Vec2 = ctx.mouse.delta().into();
            self.pos -= x / self.scale;
        }

        if let Some(rx) = &mut self.mesh_producer {
            match rx.try_recv() {
                Ok(mesh) => self.mesh = graphics::Mesh::from_data(ctx, mesh.build()),
                Err(mpsc::TryRecvError::Empty) => (),
                Err(mpsc::TryRecvError::Disconnected) => {
                    log::warn!("Mesh producer disconnected");
                    self.mesh_producer = None
                }
            }
        }

        if self.mesh_producer.is_none()
            && ctx
                .keyboard
                .is_key_just_pressed(input::keyboard::KeyCode::R)
        {
            self.mesh_producer = Some(spawn_mesh_builder(self.wfc.clone()))
        }

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> anyhow::Result<()> {
        let mut canvas =
            graphics::Canvas::from_frame(ctx, graphics::Color::from([0.92, 0.9, 0.91, 1.0]));

        canvas.draw(
            &self.mesh,
            graphics::DrawParam::new()
                .offset(self.pos)
                .scale(Vec2::splat(self.scale)),
        );

        let mut text = graphics::Text::new("Press R to restart task");

        text.set_bounds(canvas.screen_coordinates().unwrap().size())
            .set_layout(graphics::TextLayout {
                h_align: graphics::TextAlign::Begin,
                v_align: graphics::TextAlign::Begin,
            });

        canvas.draw(
            &text,
            graphics::DrawParam::default().dest_rect(canvas.screen_coordinates().unwrap()),
        );

        canvas.finish(ctx)?;

        Ok(())
    }

    fn mouse_wheel_event(&mut self, _ctx: &mut Context, _x: f32, y: f32) -> anyhow::Result<()> {
        self.scale *= 1. + y / 10.;
        Ok(())
    }
}

pub fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cb = ggez::ContextBuilder::new("rexvis", "aleok")
        .window_mode(WindowMode::default().dimensions(1080., 720.))
        .window_setup(WindowSetup::default().title("Rex Visualization & Generation Tool"));
    let (mut ctx, event_loop) = cb.build()?;
    let state = MainState::new(
        &std::env::args()
            .nth(1)
            .map(|arg| PathBuf::from(arg))
            .unwrap_or_else(|| std::env::current_dir().unwrap().to_owned()),
        &mut ctx,
    )?;
    event::run(ctx, event_loop, state)
}
