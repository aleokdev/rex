use std::path::Path;

fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-changed=app/res/basic.frag");
    println!("cargo:rerun-if-changed=app/res/basic.vert");

    let shader_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("app/res");
    let mut shader_compiler = std::process::Command::new("sh")
        .arg(shader_path.join("compile_shaders.sh"))
        .current_dir(shader_path)
        .spawn()?;

    if shader_compiler.wait()?.success() {
        Ok(())
    } else {
        panic!("Could not compile shaders")
    }
}
