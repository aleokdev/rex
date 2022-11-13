//! The simplest possible example that does something.
#![allow(clippy::unnecessary_wraps)]

use std::path::PathBuf;

use ggez::{conf::WindowMode, event, glam::*, graphics, input, Context};
use rex::Room;

struct MainState {
    pos: Vec2,
    scale: f32,
    mesh: graphics::Mesh,
    room: rex::Room,
}

impl MainState {
    fn new(path: &std::path::Path, ctx: &mut Context) -> anyhow::Result<MainState> {
        let room = rex::generate_v2(path)?;
        let mut builder = graphics::MeshBuilder::new();
        build_mesh(&mut builder, &room)?;

        Ok(MainState {
            pos: Default::default(),
            scale: 10.,
            room,
            mesh: graphics::Mesh::from_data(ctx, builder.build()),
        })
    }
}

fn build_mesh(builder: &mut graphics::MeshBuilder, room: &Room) -> anyhow::Result<()> {
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
            0.05,
            graphics::Color::from_rgba_u32(0xFFFFFF50),
        )?;
        build_mesh(builder, child)?;
    }

    Ok(())
}

impl event::EventHandler<anyhow::Error> for MainState {
    fn update(&mut self, ctx: &mut Context) -> anyhow::Result<()> {
        if ctx.mouse.button_pressed(input::mouse::MouseButton::Middle) {
            let x: Vec2 = ctx.mouse.delta().into();
            self.pos -= x / self.scale;
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
