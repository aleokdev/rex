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
}

impl MainState {
    fn new(path: &std::path::Path, ctx: &mut Context) -> anyhow::Result<MainState> {
        let mut nodes = rex::generate_nodes(path)?;
        let (tx, rx) = mpsc::channel();
        let path = path.to_owned();
        thread::spawn(move || {
            let start = std::time::Instant::now();
            let mut fdd = rex::ForceDirectedDrawing::new(&mut nodes);

            loop {
                match fdd.iterate() {
                    ControlFlow::Break(_) => break,
                    ControlFlow::Continue(loss) => {
                        let mut builder = graphics::MeshBuilder::new();
                        build_nodes_mesh(&mut builder, fdd.nodes());
                        tx.send(builder).unwrap();
                        log::info!("T: {}, Loss: {}", fdd.t(), loss)
                    }
                }
            }
            log::info!(
                "Took {}s in total",
                (std::time::Instant::now() - start).as_secs_f32()
            );

            let room = rex::nodes_to_room(nodes, &path);
            let mut builder = graphics::MeshBuilder::new();
            build_room_mesh(&mut builder, &room);
            tx.send(builder);
        });

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
        })
    }
}

fn build_nodes_mesh(
    builder: &mut graphics::MeshBuilder,
    nodes: &Vec<rex::Node>,
) -> anyhow::Result<()> {
    for node in nodes.iter() {
        builder.circle(
            graphics::DrawMode::Fill(graphics::FillOptions::default()),
            node.pos(),
            10.,
            0.01,
            graphics::Color::RED,
        )?;
    }

    Ok(())
}

fn build_room_mesh(builder: &mut graphics::MeshBuilder, room: &Room) -> anyhow::Result<()> {
    builder.circle(
        graphics::DrawMode::Fill(graphics::FillOptions::default()),
        room.mesh[0],
        0.1,
        0.01,
        graphics::Color::RED,
    )?;
    for child in &room.children {
        builder.line(
            &[room.mesh[0], child.mesh[0]],
            3.,
            graphics::Color::from_rgba_u32(0xFFFFFF50),
        )?;
        build_room_mesh(builder, child)?;
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
                Err(mpsc::TryRecvError::Disconnected) => self.mesh_producer = None,
            }
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
                h_align: graphics::TextAlign::End,
                v_align: graphics::TextAlign::End,
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
    let cb = ggez::ContextBuilder::new("super_simple", "ggez")
        .window_mode(WindowMode::default().dimensions(1920., 1080.));
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
