mod app;
mod buffer;
mod cx;
mod image;
mod memory;
mod util;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    app::run(800, 600)
}
