//! The simplest possible example that does something.
#![allow(clippy::unnecessary_wraps)]

use std::{ops::ControlFlow, path::PathBuf, sync::mpsc, thread, time::Duration};

use ggez::{
    conf::{WindowMode, WindowSetup},
    event,
    glam::*,
    graphics, input, Context,
};
use rex::Room;

struct MainState {
    pos: Vec2,
    scale: f32,
    mesh: graphics::Mesh,
    mesh_producer: Option<mpsc::Receiver<graphics::MeshBuilder>>,
    nodes: Vec<rex::Node>,
}

impl MainState {
    fn new(path: &std::path::Path, ctx: &mut Context) -> anyhow::Result<MainState> {
        let nodes = rex::generate_nodes(path)?;
        let rx = spawn_mesh_builder(nodes.clone());

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
            nodes,
        })
    }
}

fn spawn_mesh_builder(nodes: Vec<rex::Node>) -> mpsc::Receiver<graphics::MeshBuilder> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let start = std::time::Instant::now();

        let mut v4 = rex::V4::new(nodes);
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

        let (nodes, rooms, space) = v4.build();
        let mut v4c = rex::V4CorridorSolver::new(nodes, rooms, space);
        let mut iterations = 0;

        loop {
            match v4c.iterate() {
                ControlFlow::Break(_) => break,
                ControlFlow::Continue(_) => {}
            }

            if iterations % 10 == 0 {
                let mut builder = graphics::MeshBuilder::new();
                build_room_mesh(&mut builder, v4c.rooms()).unwrap();
                build_paths_mesh(&mut builder, v4c.paths()).unwrap();
                tx.send(builder).unwrap();
            }

            iterations += 1;
        }
        log::info!(
            "Took {}s in total",
            (std::time::Instant::now() - start).as_secs_f32()
        );

        let (nodes, rooms, paths, space) = v4c.build();
        let mut v4s = rex::V4CorridorSmoother::new(paths, space);

        loop {
            match v4s.iterate() {
                ControlFlow::Break(_) => break,
                ControlFlow::Continue(_) => (),
            }

            if iterations % 10 == 0 {
                let mut builder = graphics::MeshBuilder::new();
                build_room_mesh(&mut builder, &rooms).unwrap();
                build_paths_mesh(&mut builder, v4s.paths()).unwrap();
                tx.send(builder).unwrap();
            }

            iterations += 1;
        }
        log::info!(
            "Took {}s in total",
            (std::time::Instant::now() - start).as_secs_f32()
        );

        let mut builder = graphics::MeshBuilder::new();
        build_room_mesh(&mut builder, &rooms).unwrap();
        build_paths_mesh(&mut builder, &v4s.paths()).unwrap();
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

fn build_paths_mesh(
    builder: &mut graphics::MeshBuilder,
    paths: &[Vec<mint::Vector2<f32>>],
) -> anyhow::Result<()> {
    for path in paths {
        if path.len() > 2 {
            builder.polyline(
                graphics::DrawMode::Stroke(graphics::StrokeOptions::default().with_line_width(0.2)),
                &path,
                graphics::Color::GREEN,
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
            self.mesh_producer = Some(spawn_mesh_builder(self.nodes.clone()))
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
