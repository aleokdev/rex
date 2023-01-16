//! The simplest possible example that does something.
#![allow(clippy::unnecessary_wraps)]

use std::{collections::HashMap, ops::ControlFlow, path::PathBuf, sync::mpsc, thread};

use ggez::{
    conf::{WindowMode, WindowSetup},
    event,
    glam::*,
    graphics, input, Context,
};
use rex::{grid::RoomId, node::Node, space::SpaceAllocation, Door, Wall};

enum MeshProducerData {
    OnlyFloor0(graphics::MeshBuilder),
    All {
        database: rex::Database,
        meshes: HashMap<i32, graphics::MeshBuilder>,
    },
}

struct MainState {
    pos: Vec2,
    scale: f32,
    meshes: HashMap<i32, graphics::Mesh>,
    mesh_producer: Option<mpsc::Receiver<MeshProducerData>>,
    nodes: Vec<rex::node::Node>,
    current_floor: i32,
    database: Option<rex::Database>,
}

impl MainState {
    fn new(path: &std::path::Path, ctx: &mut Context) -> anyhow::Result<MainState> {
        let nodes = rex::node::generate_nodes(path)?;
        let rx = spawn_mesh_builder(nodes.clone());

        Ok(MainState {
            pos: vec2(-512., -512.),
            scale: 1.,
            meshes: HashMap::new(),
            mesh_producer: Some(rx),
            nodes,
            current_floor: 0,
            database: None,
        })
    }
}

fn spawn_mesh_builder(nodes: Vec<rex::node::Node>) -> mpsc::Receiver<MeshProducerData> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let start = std::time::Instant::now();

        let mut v3 = rex::V3::new(nodes.clone());
        let mut rng = rand::thread_rng();
        let mut iterations = 0;

        let v3data = loop {
            match v3.iterate(&mut rng) {
                ControlFlow::Break(database) => break database,
                ControlFlow::Continue(cont) => v3 = cont,
            }

            if iterations % 500 == 0 {
                let mut builder = graphics::MeshBuilder::new();
                build_room_mesh(
                    &mut builder,
                    v3.database()
                        .map
                        .floor(0)
                        .unwrap()
                        .grid()
                        .cells()
                        .filter_map(|(pos, &cell)| {
                            cell.map(|id| (ggez::glam::IVec2::new(pos.x, pos.y), id))
                        }),
                )
                .unwrap();
                tx.send(MeshProducerData::OnlyFloor0(builder)).unwrap();
            }

            iterations += 1;
        };
        log::info!(
            "Total stats: Teleports used: {}. Took {:.3}s.",
            v3data.teleports_used,
            (std::time::Instant::now() - start).as_secs_f32()
        );
        tx.send(MeshProducerData::All {
            meshes: v3data
                .database
                .map
                .floors()
                .map(|(&floor_idx, map)| {
                    let walls = rex::generate_wall_map(map.grid());
                    let mut builder = graphics::MeshBuilder::new();
                    build_room_mesh(
                        &mut builder,
                        map.grid().cells().filter_map(|(pos, &cell)| {
                            cell.map(|id| (ggez::glam::IVec2::new(pos.x, pos.y), id))
                        }),
                    )
                    .unwrap();
                    build_room_mesh_walls(
                        &mut builder,
                        walls
                            .cells()
                            .map(|(pos, &cell)| (ggez::glam::IVec2::new(pos.x, pos.y), cell)),
                    )
                    .unwrap();

                    (floor_idx, builder)
                })
                .collect(),
            database: v3data.database,
        })
        .unwrap();
    });
    rx
}

fn build_allocator_mesh<'s>(
    builder: &mut graphics::MeshBuilder,
    allocations: impl Iterator<Item = &'s SpaceAllocation>,
) -> anyhow::Result<()> {
    for alloc in allocations {
        builder.circle(
            graphics::DrawMode::stroke(0.3),
            alloc.pos,
            alloc.radius,
            0.1,
            graphics::Color::from_rgba(255, 0, 0, 100),
        )?;
    }

    Ok(())
}
fn build_network_mesh<'s>(
    builder: &mut graphics::MeshBuilder,
    nodes: &[Node],
    node_positions: &[rex::glam::IVec2],
) -> anyhow::Result<()> {
    for (id, node) in nodes.iter().enumerate() {
        let start = node_positions[id];
        let start = vec2(start.x as f32, start.y as f32);
        for &child in node.children.iter() {
            let end = node_positions[child];
            let end = vec2(end.x as f32, end.y as f32);
            builder.line(&[start, end], 0.3, graphics::Color::BLUE)?;
        }
    }

    Ok(())
}

fn build_room_mesh(
    builder: &mut graphics::MeshBuilder,
    cells: impl Iterator<Item = (IVec2, RoomId)>,
) -> anyhow::Result<()> {
    for (pos, id) in cells {
        builder.rectangle(
            graphics::DrawMode::fill(),
            ggez::graphics::Rect::new(pos.x as f32, pos.y as f32, 1., 1.),
            graphics::Color::from_rgb(
                ((id * 163) % 256) as u8,
                ((id * 483) % 256) as u8,
                ((id * 773) % 256) as u8,
            ),
        )?;
    }

    Ok(())
}

fn build_room_mesh_walls(
    builder: &mut graphics::MeshBuilder,
    cells: impl Iterator<Item = (IVec2, Wall)>,
) -> anyhow::Result<()> {
    for (pos, wall) in cells {
        let x = pos.x as f32;
        let y = pos.y as f32;
        const WALL_WIDTH: f32 = 0.1;
        let segments = [
            // North
            graphics::Rect::new(x + WALL_WIDTH, y, 1.0 - WALL_WIDTH * 2., WALL_WIDTH),
            // East
            graphics::Rect::new(
                x + 1.0 - WALL_WIDTH,
                y + WALL_WIDTH,
                WALL_WIDTH,
                1.0 - WALL_WIDTH * 2.,
            ),
            // West
            graphics::Rect::new(x, y + WALL_WIDTH, WALL_WIDTH, 1.0 - WALL_WIDTH * 2.),
            // South
            graphics::Rect::new(
                x + WALL_WIDTH,
                y + 1.0 - WALL_WIDTH,
                1.0 - WALL_WIDTH * 2.,
                WALL_WIDTH,
            ),
            // Northwest
            graphics::Rect::new(x, y, WALL_WIDTH, WALL_WIDTH),
            // Northeast
            graphics::Rect::new(x + 1.0 - WALL_WIDTH, y, WALL_WIDTH, WALL_WIDTH),
            // Southwest
            graphics::Rect::new(x, y + 1.0 - WALL_WIDTH, WALL_WIDTH, WALL_WIDTH),
            // Southeast
            graphics::Rect::new(
                x + 1.0 - WALL_WIDTH,
                y + 1.0 - WALL_WIDTH,
                WALL_WIDTH,
                WALL_WIDTH,
            ),
        ];
        for segment_idx in 0..8 {
            if wall.contains(Wall::from_bits_truncate(1 << segment_idx)) {
                let segment = segments[segment_idx];
                builder.rectangle(graphics::DrawMode::fill(), segment, graphics::Color::BLACK)?;
            }
        }
    }

    Ok(())
}

fn build_room_mesh_doors<'s>(
    builder: &mut graphics::MeshBuilder,
    doors: impl Iterator<Item = &'s Door>,
) -> anyhow::Result<()> {
    for door in doors {
        let start_point = vec2(door.position.x, door.position.y);
        let end_point = start_point + vec2(door.rotation.cos(), door.rotation.sin());
        builder.line(&[start_point, end_point], 0.2, graphics::Color::BLUE)?;
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
                Ok(MeshProducerData::OnlyFloor0(floor0)) => {
                    self.meshes =
                        HashMap::from_iter([(0, graphics::Mesh::from_data(ctx, floor0.build()))])
                }
                Ok(MeshProducerData::All { meshes, database }) => {
                    self.meshes = HashMap::from_iter(meshes.iter().map(|(&floor_idx, builder)| {
                        (floor_idx, graphics::Mesh::from_data(ctx, builder.build()))
                    }));
                    self.database = Some(database);
                }
                Err(mpsc::TryRecvError::Empty) => (),
                Err(mpsc::TryRecvError::Disconnected) => {
                    log::warn!("Mesh producer disconnected");
                    self.mesh_producer = None
                }
            }
        }

        if ctx
            .keyboard
            .is_key_just_pressed(input::keyboard::KeyCode::R)
        {
            self.mesh_producer = Some(spawn_mesh_builder(self.nodes.clone()))
        }

        if ctx
            .keyboard
            .is_key_just_pressed(input::keyboard::KeyCode::Up)
        {
            self.current_floor += 1;
        }
        if ctx
            .keyboard
            .is_key_just_pressed(input::keyboard::KeyCode::Down)
        {
            self.current_floor -= 1;
        }
        if let Some(database) = &self.database {
            if ctx
                .keyboard
                .is_key_just_pressed(input::keyboard::KeyCode::S)
            {
                match nfd::open_save_dialog(Some("rex"), None) {
                    Ok(nfd::Response::Okay(file)) => {
                        serde_json::to_writer(std::fs::File::create(file)?, database)?
                    }
                    Err(x) => log::error!("{}", x),
                    _ => (),
                }
            }
        }

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> anyhow::Result<()> {
        let mut canvas =
            graphics::Canvas::from_frame(ctx, graphics::Color::from([0.12, 0.0, 0.21, 1.0]));

        let mut white = 55;
        for floor_idx in self.current_floor - 1..=self.current_floor {
            if let Some(floor_mesh) = self.meshes.get(&floor_idx) {
                canvas.draw(
                    floor_mesh,
                    graphics::DrawParam::new()
                        .offset(self.pos)
                        .scale(Vec2::splat(self.scale))
                        .color(graphics::Color::from_rgb(white, white, white)),
                );
            }
            if white < 255 {
                white += 200;
            }
        }

        let mut text = graphics::Text::new({
            let mut text = format!(
                "Top-left corner: {:.0}\nCurrent floor:{}",
                self.pos, self.current_floor
            );
            if self.database.is_some() {
                text += "Database available, press S to save";
            }
            text
        });

        text.set_bounds(Vec2::new(500.0, f32::INFINITY))
            .set_layout(graphics::TextLayout {
                h_align: graphics::TextAlign::Begin,
                v_align: graphics::TextAlign::Begin,
            });

        canvas.draw(&text, graphics::DrawParam::default());

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
