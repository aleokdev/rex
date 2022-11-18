//! The simplest possible example that does something.
#![allow(clippy::unnecessary_wraps)]

use std::{ops::ControlFlow, path::PathBuf, sync::mpsc, thread};

use ggez::{conf::WindowMode, event, glam::*, graphics, input, Context};
use rex::Room;

struct MainState {
    pos: Vec2,
    scale: f32,
    mesh: graphics::Mesh,
    mesh_producer: Option<mpsc::Receiver<graphics::MeshBuilder>>,
    nodes: Vec<rex::Node>,
    radius: f32,
}

impl MainState {
    fn new(path: &std::path::Path, ctx: &mut Context) -> anyhow::Result<MainState> {
        let nodes = rex::generate_nodes(path)?;
        let radius = 5.;
        let rx = spawn_mesh_builder(nodes.clone(), radius);

        Ok(MainState {
            pos: vec2(-512., -512.),
            scale: 1.,
            mesh: graphics::Mesh::from_data(
                ctx,
                graphics::MeshData {
                    vertices: &[],
                    indices: &[],
                },
            ),
            mesh_producer: Some(rx),
            nodes,
            radius,
        })
    }
}

fn spawn_mesh_builder(nodes: Vec<rex::Node>, radius: f32) -> mpsc::Receiver<graphics::MeshBuilder> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let start = std::time::Instant::now();
        let mut db = rex::RoomTemplateDb::default();
        db.push({
            rex::grid::RoomTemplate::with_inscribed_radius_and_outline(
                {
                    let mut grid = rex::grid::CartesianGrid::default();
                    grid.set_cell(ivec2(-1, -1), rex::grid::RoomTemplateCell::Occupied);
                    grid.set_cell(ivec2(-1, 0), rex::grid::RoomTemplateCell::Occupied);
                    grid.set_cell(ivec2(-1, 1), rex::grid::RoomTemplateCell::Occupied);
                    grid.set_cell(ivec2(0, -1), rex::grid::RoomTemplateCell::Occupied);
                    grid.set_cell(ivec2(0, 0), rex::grid::RoomTemplateCell::Occupied);
                    grid.set_cell(ivec2(0, 1), rex::grid::RoomTemplateCell::Occupied);
                    grid.set_cell(ivec2(1, -1), rex::grid::RoomTemplateCell::Occupied);
                    grid.set_cell(ivec2(1, 0), rex::grid::RoomTemplateCell::Occupied);
                    grid.set_cell(ivec2(1, 1), rex::grid::RoomTemplateCell::Occupied);
                    grid
                },
                radius,
                &[
                    vec2(-1.5, -1.5),
                    vec2(-1.5, 1.5),
                    vec2(1.5, 1.5),
                    vec2(1.5, -1.5),
                ],
            )
        });

        let mut v4 = rex::V4::new(nodes, radius);
        let mut rng = rand::thread_rng();
        let mut iterations = 0;

        loop {
            match v4.iterate(&mut rng) {
                ControlFlow::Break(_) => break,
                ControlFlow::Continue(_) => {}
            }

            if iterations % 100 == 0 {
                let mut builder = graphics::MeshBuilder::new();
                build_room_mesh(&mut builder, v4.rooms()).unwrap();
                tx.send(builder).unwrap();
            }

            iterations += 1;
        }
        log::info!(
            "Took {}s in total",
            (std::time::Instant::now() - start).as_secs_f32()
        );

        let mut builder = graphics::MeshBuilder::new();
        build_room_mesh(&mut builder, &v4.rooms()).unwrap();
        tx.send(builder);
    });
    rx
}

fn build_room_mesh(builder: &mut graphics::MeshBuilder, rooms: &[Room]) -> anyhow::Result<()> {
    for room in rooms {
        if room.mesh.len() > 0 {
            builder.polyline(
                graphics::DrawMode::Fill(graphics::FillOptions::default()),
                &room.mesh,
                graphics::Color::RED,
            )?;
            builder.polygon(
                graphics::DrawMode::Stroke(graphics::StrokeOptions::default().with_line_width(0.1)),
                &room.mesh,
                graphics::Color::WHITE,
            )?;
        }
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
            self.mesh_producer = Some(spawn_mesh_builder(self.nodes.clone(), self.radius))
        }

        if ctx
            .keyboard
            .is_key_just_pressed(input::keyboard::KeyCode::Up)
        {
            self.radius += 1.;
        }
        if ctx
            .keyboard
            .is_key_just_pressed(input::keyboard::KeyCode::Down)
        {
            self.radius -= 1.;
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
    let cb = ggez::ContextBuilder::new("super_simple", "ggez")
        .window_mode(WindowMode::default().dimensions(1080., 720.));
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
