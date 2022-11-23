fn main() -> anyhow::Result<()> {
    env_logger::init();
    app::run(800, 600)
}
