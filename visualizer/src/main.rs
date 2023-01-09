//! The simplest possible example that does something.
#![allow(clippy::unnecessary_wraps)]

use std::{ops::ControlFlow, path::PathBuf, sync::mpsc, thread};

use ggez::{
    conf::{WindowMode, WindowSetup},
    event,
    glam::*,
    graphics::{self, DrawMode, Rect},
    input, Context,
};
use rand::seq::IteratorRandom;
use rex::{
    wfc::{self, Tile},
    Room,
};

struct MainState {
    pos: Vec2,
    scale: f32,
    mesh: graphics::Mesh,
    mesh_producer: Option<mpsc::Receiver<graphics::MeshBuilder>>,
    wfc: wfc::Context,
}

impl MainState {
    fn new(path: &std::path::Path, ctx: &mut Context) -> anyhow::Result<MainState> {
        let wfc = wfc::Context::new(vec![
            Tile::from_rules(wfc::SpatialRuleMap::from_fn(|pos| {
                match (pos.x, pos.y, pos.z) {
                    (1, 0, 0) => [0, 1].into_iter().collect(),
                    (-1, 0, 0) => [0, 1].into_iter().collect(),
                    (0, 0, _) => [0, 1, 2, 3].into_iter().collect(),
                    (x, _, z) if x != 0 && z != 0 => [0, 1, 2, 3].into_iter().collect(),
                    _ => Default::default(),
                }
            })),
            Tile::from_rules(wfc::SpatialRuleMap::from_fn(|pos| {
                match (pos.x, pos.y, pos.z) {
                    (1, 0, 0) => [0, 1].into_iter().collect(),
                    (-1, 0, 0) => [0, 1].into_iter().collect(),
                    (0, 0, 1) => [3].into_iter().collect(),
                    (x, _, z) if x != 0 && z != 0 => [0, 1, 2, 3].into_iter().collect(),
                    _ => Default::default(),
                }
            })),
            Tile::from_rules(wfc::SpatialRuleMap::from_fn(|_pos| {
                [0, 1, 2, 3].into_iter().collect()
            })),
            Tile::from_rules(wfc::SpatialRuleMap::from_fn(|pos| {
                match (pos.x, pos.y, pos.z) {
                    (0, 0, 1) => [3].into_iter().collect(),
                    (0, 0, -1) => [3, 1].into_iter().collect(),
                    (_, 0, 0) => [0, 1, 2, 3].into_iter().collect(),
                    (x, _, z) if x != 0 && z != 0 => [0, 1, 2, 3].into_iter().collect(),
                    _ => Default::default(),
                }
            })),
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
        world.place(&ctx, 1, glam::ivec3(2, 0, 3)).unwrap();
        for x in -10..=10 {
            world.place(&ctx, 2, glam::ivec3(x, 0, 10)).unwrap();
            world.place(&ctx, 2, glam::ivec3(x, 0, -10)).unwrap();
        }
        for z in -9..=9 {
            world.place(&ctx, 2, glam::ivec3(10, 0, z)).unwrap();
            world.place(&ctx, 2, glam::ivec3(-10, 0, z)).unwrap();
        }
        dbg!(&world.space[&glam::ivec3(0, 0, 9)]);
        let mut rng = rand::thread_rng();

        let mut builder = graphics::MeshBuilder::new();
        build_world_mesh(&mut builder, &world).unwrap();
        tx.send(builder).unwrap();
        thread::sleep(std::time::Duration::from_millis(1000));
        let start = std::time::Instant::now();

        for _ in 0..1000 {
            let random_min_entropy_pos = match world.least_entropy_positions_2d() {
                Some(x) => x.choose(&mut rng).unwrap(),
                None => glam::IVec3::ZERO,
            };
            world
                .collapse(&ctx, &mut rng, random_min_entropy_pos)
                .unwrap();
            let mut builder = graphics::MeshBuilder::new();
            build_world_mesh(&mut builder, &world).unwrap();
            tx.send(builder).unwrap();
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
        log::info!("{}", pos);
        match tile {
            &wfc::SpaceTile::Collapsed(index) => match index {
                0 => {
                    builder.rectangle(
                        DrawMode::fill(),
                        Rect::new(pos.x as f32, pos.z as f32, 1., 1.),
                        ggez::graphics::Color::RED,
                    )?;
                    builder.line(
                        &[
                            [pos.x as f32, pos.z as f32 + 0.5],
                            [pos.x as f32 + 1.0, pos.z as f32 + 0.5],
                        ],
                        0.3,
                        graphics::Color::BLACK,
                    )?
                }
                1 => {
                    builder.rectangle(
                        DrawMode::fill(),
                        Rect::new(pos.x as f32, pos.z as f32, 1., 1.),
                        ggez::graphics::Color::BLUE,
                    )?;

                    builder.line(
                        &[
                            [pos.x as f32, pos.z as f32 + 0.5],
                            [pos.x as f32 + 1.0, pos.z as f32 + 0.5],
                        ],
                        0.3,
                        graphics::Color::BLACK,
                    )?;

                    builder.line(
                        &[
                            [pos.x as f32 + 0.5, pos.z as f32 + 0.5],
                            [pos.x as f32 + 0.5, pos.z as f32 + 1.0],
                        ],
                        0.3,
                        graphics::Color::BLACK,
                    )?
                }
                2 => builder.rectangle(
                    DrawMode::fill(),
                    Rect::new(pos.x as f32, pos.z as f32, 1., 1.),
                    ggez::graphics::Color::GREEN,
                )?,
                3 => {
                    builder.rectangle(
                        DrawMode::fill(),
                        Rect::new(pos.x as f32, pos.z as f32, 1., 1.),
                        ggez::graphics::Color::YELLOW,
                    )?;

                    builder.line(
                        &[
                            [pos.x as f32 + 0.5, pos.z as f32],
                            [pos.x as f32 + 0.5, pos.z as f32 + 1.0],
                        ],
                        0.3,
                        graphics::Color::BLACK,
                    )?
                }

                _ => panic!(),
            },
            wfc::SpaceTile::NonCollapsed(h) => {
                let mut color = ggez::graphics::Color::WHITE;
                color.a = (h.len() as f32 / 3.0).min(1.0);
                builder.rectangle(
                    DrawMode::fill(),
                    Rect::new(pos.x as f32, pos.z as f32, 1., 1.),
                    color,
                )?
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
            graphics::Canvas::from_frame(ctx, graphics::Color::from([0.12, 0.0, 0.21, 1.0]));

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
