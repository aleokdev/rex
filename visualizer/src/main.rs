//! The simplest possible example that does something.
#![allow(clippy::unnecessary_wraps)]

use std::{collections::HashMap, ops::ControlFlow, path::PathBuf, sync::mpsc, thread};

use ggez::{
    conf::{WindowMode, WindowSetup},
    event,
    graphics::{self, Rect},
    input, Context,
};
use rex::{
    building::{DualNormalDirection, Room},
    glam::*,
    grid::RoomId,
    node::Node,
    space::SpaceAllocation,
    Database, Door, Wall,
};

enum MeshProducerData {
    OnlyFloor0(graphics::MeshBuilder),
    All {
        database: rex::Database,
        meshes: HashMap<i32, graphics::MeshBuilder>,
    },
}

struct MainState {
    pos: Vec2,
    proj_width: f32,
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
            pos: vec2(0., 0.),
            proj_width: 50.,
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
                        .filter_map(|(pos, &cell)| cell.map(|id| (IVec2::new(pos.x, pos.y), id))),
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
                            cell.map(|id| (IVec2::new(pos.x, pos.y), id))
                        }),
                    )
                    .unwrap();
                    map.room_cell_positions().keys().for_each(|&room_id| {
                        build_room_mesh_walls(&mut builder, &v3data.database.rooms[room_id])
                            .unwrap()
                    });

                    (floor_idx, builder)
                })
                .collect(),
            database: v3data.database,
        })
        .unwrap();
    });
    rx
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

fn build_room_mesh_walls(builder: &mut graphics::MeshBuilder, room: &Room) -> anyhow::Result<()> {
    for (dual_pos, piece) in room.duals.iter() {
        const WALL_WIDTH: f32 = 0.1;
        let is_horizontal = (dual_pos.x + dual_pos.y) % 2 == 0;
        let corrected_dual_pos = if is_horizontal {
            dual_pos.truncate()
        } else {
            dual_pos.truncate() - IVec2::Y
        };
        let cell_pos = ivec2(
            (corrected_dual_pos.x - corrected_dual_pos.y) / 2,
            (corrected_dual_pos.x + corrected_dual_pos.y) / 2,
        );
        let mut x = cell_pos.x as f32;
        let mut y = cell_pos.y as f32;
        match piece {
            rex::building::DualPiece::Wall { normal }
            | rex::building::DualPiece::Door { normal } => {
                if *normal == DualNormalDirection::NorthWest {
                    x -= WALL_WIDTH;
                    y -= WALL_WIDTH;
                }
            }
        }
        builder.rectangle(
            graphics::DrawMode::fill(),
            if is_horizontal {
                graphics::Rect::new(x + WALL_WIDTH, y, 1.0 - WALL_WIDTH * 2., WALL_WIDTH)
            } else {
                graphics::Rect::new(x, y + WALL_WIDTH, WALL_WIDTH, 1.0 - WALL_WIDTH * 2.)
            },
            graphics::Color::MAGENTA,
        )?;
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
            let window_size = ctx.gfx.window().inner_size();
            self.pos -= x * self.proj_width / window_size.width as f32;
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
            self.mesh_producer = Some(spawn_mesh_builder(self.nodes.clone()));
            self.database = None;
            self.meshes.clear();
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
                    Ok(nfd::Response::Okay(file)) => serde_json::to_writer(
                        std::io::BufWriter::new(std::fs::File::create(file)?),
                        database,
                    )?,
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

        let window_size = ctx.gfx.window().inner_size();
        let aspect_ratio = window_size.width as f32 / window_size.height as f32;
        let half_width = self.proj_width / 2.;
        canvas.set_screen_coordinates(Rect::new(
            self.pos.x - half_width,
            self.pos.y - half_width / aspect_ratio,
            self.proj_width,
            self.proj_width / aspect_ratio,
        ));

        canvas.draw(
            &graphics::Quad,
            graphics::DrawParam::new().dest_rect(Rect::new(-0.5, -0.5, 1., 1.)),
        );
        let mut white = 55;
        for floor_idx in self.current_floor - 1..=self.current_floor {
            if let Some(floor_mesh) = self.meshes.get(&floor_idx) {
                canvas.draw(
                    floor_mesh,
                    graphics::DrawParam::new()
                        .color(graphics::Color::from_rgb(white, white, white)),
                );
            }
            if white < 255 {
                white += 200;
            }
        }

        let mut text = graphics::Text::new({
            let mut text = format!(
                "Screen center: {:.0}\nCurrent floor:{}",
                self.pos, self.current_floor
            );
            if self.database.is_some() {
                text += "\nDatabase available, press S to save";
            }
            text
        });
        if let Some(database) = &self.database {
            if let Some(floor) = database.map.floor(self.current_floor) {
                let screen_coords = canvas.screen_coordinates().unwrap();
                let mouse_window_pos = Vec2::from(ctx.mouse.position());
                let to_world_coords = |x: Vec2| -> Vec2 {
                    x / Vec2::new(window_size.width as f32, window_size.height as f32)
                        * Vec2::from(screen_coords.size())
                        + Vec2::from(screen_coords.point())
                };
                let to_window_coords = |x: Vec2| -> Vec2 {
                    (x - Vec2::from(screen_coords.point())) / Vec2::from(screen_coords.size())
                        * Vec2::new(window_size.width as f32, window_size.height as f32)
                };
                let mouse_world_pos = to_world_coords(mouse_window_pos);
                let mouse_world_pos = mouse_world_pos.as_ivec2();

                text.add(format!("\n{}", mouse_world_pos));

                if let Some(room_id_below_cursor) = floor.cell(mouse_world_pos) {
                    let room_below_cursor = &database.rooms[room_id_below_cursor];
                    let node = &database.nodes[room_below_cursor.node()];
                    let hover_text = graphics::Text::new(format!(
                        "path: {:?}\nroom id: {}\nnode id: {}\nnode parent: {:?}\nnode children: {}",
                        node.path,
                        room_id_below_cursor,
                        room_below_cursor.node(),
                        node.parent,
                        if node.children.len() > 3 { format!("{:?} along others", &node.children[..3])} else { format!("{:?}", &node.children[..])}
                    ));

                    canvas.draw(
                        &hover_text,
                        graphics::DrawParam::default().dest(mouse_window_pos),
                    );

                    let start_pos = database.rooms[room_id_below_cursor].starting_pos();
                    let mut builder = graphics::MeshBuilder::new();
                    if let Some(parent) = node.parent {
                        let end_pos = database.rooms
                            [*database.nodes[parent].rooms.first().unwrap()]
                        .starting_pos();
                        builder.line(
                            &[start_pos.truncate().as_vec2(), end_pos.truncate().as_vec2()],
                            self.proj_width / 100.,
                            graphics::Color::RED,
                        )?;
                    }
                    for &child in node.children.iter() {
                        let end_pos = database.rooms[*database.nodes[child].rooms.first().unwrap()]
                            .starting_pos();
                        if end_pos.z != start_pos.z {
                            continue;
                        }
                        builder.line(
                            &[start_pos.truncate().as_vec2(), end_pos.truncate().as_vec2()],
                            self.proj_width / 100.,
                            graphics::Color::GREEN,
                        )?;
                    }
                    canvas.draw(
                        &graphics::Mesh::from_data(&ctx.gfx, builder.build()),
                        graphics::DrawParam::default(),
                    );
                }
            }
        }

        canvas.set_screen_coordinates(Rect::new(
            0.,
            0.,
            window_size.width as f32,
            window_size.height as f32,
        ));

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
        self.proj_width *= 1. - y / 10.;
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
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap()),
        &mut ctx,
    )?;
    event::run(ctx, event_loop, state)
}
