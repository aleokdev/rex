mod collision;
mod meshgen;

use std::collections::HashSet;

use glam::{ivec3, vec3, Vec2, Vec3};
use renderer::{
    abs::mesh::{self, CpuMesh},
    Camera, RenderData,
};
use rex::grid::RoomId;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::CursorGrabMode,
};

use crate::{collision::move_and_slide, meshgen::CEILING_HEIGHT};

struct App {
    pub cx: renderer::abs::Cx,
    pub renderer: renderer::Renderer,
    pub render_data: RenderData,
    room_meshes_uploaded: HashSet<RoomId>,
    database: rex::Database,
}

impl App {
    unsafe fn new(event_loop: &EventLoop<()>, width: u32, height: u32) -> anyhow::Result<Self> {
        let mut cx = renderer::abs::Cx::new(event_loop, width, height)?;
        log::info!("Loading database...");
        let start = std::time::Instant::now();
        let database = serde_json::from_reader::<_, rex::Database>(std::io::BufReader::new(
            std::fs::File::open("test.rex").unwrap(),
        ))?;
        log::info!(
            "Loaded database in {:3}s",
            (std::time::Instant::now() - start).as_secs_f32()
        );
        let mut renderer = renderer::Renderer::new(&mut cx)?;
        let mut render_data = RenderData {
            camera: Camera::new(vec3(0., 0., 1.68), cx.width as f32 / cx.height as f32),
            objects: vec![],
        };

        let mut models = tobj::load_obj(
            "app/res/xyz_gizmo.obj",
            &tobj::LoadOptions {
                ignore_lines: true,
                ignore_points: true,
                single_index: true,
                triangulate: true,
            },
        )?
        .0;
        models.extend(
            tobj::load_obj(
                "app/res/uvcube.obj",
                &tobj::LoadOptions {
                    ignore_lines: true,
                    ignore_points: true,
                    single_index: true,
                    triangulate: true,
                },
            )?
            .0
            .into_iter(),
        );

        let texture = renderer.upload_image(image::open("app/res/unknown.png")?.into_rgb8());
        for model in models {
            let mut mesh = model.mesh;
            if mesh.vertex_color.is_empty() {
                mesh.vertex_color
                    .extend(std::iter::repeat(1.).take(mesh.positions.len()));
            }
            if mesh.texcoords.is_empty() {
                mesh.texcoords
                    .extend(std::iter::repeat(0.).take(mesh.positions.len() / 3 * 2));
            }
            let vertices = mesh
                .positions
                .chunks_exact(3)
                .zip(mesh.normals.chunks_exact(3))
                .zip(mesh.vertex_color.chunks_exact(3))
                .zip(mesh.texcoords.chunks_exact(2))
                .map(|(((position, normal), color), uv)| mesh::Vertex {
                    position: <[f32; 3]>::try_from(position).unwrap().into(),
                    normal: <[f32; 3]>::try_from(normal).unwrap().into(),
                    color: <[f32; 3]>::try_from(color).unwrap().into(),
                    uv: <[f32; 2]>::try_from(uv).unwrap().into(),
                })
                .collect::<Vec<_>>();
            let indices = mesh.indices;
            render_data.objects.push(renderer::RenderObject {
                mesh_handle: renderer.upload_mesh(CpuMesh { vertices, indices }),
                material: renderer::Material::TexturedLit { texture },
            });
        }

        let mut room_meshes_uploaded = HashSet::new();
        if std::env::var("REX_MESHGEN_EVERYTHING").is_ok() {
            for room_id in 0..database.rooms.len() {
                render_data.objects.push(renderer::RenderObject {
                    mesh_handle: renderer
                        .upload_mesh(meshgen::generate_room_mesh(&database, room_id)),
                    material: renderer::Material::FlatLit,
                });
                room_meshes_uploaded.insert(room_id);
                if room_id % 100 == 0 {
                    log::info!("{}/{}", room_id, database.rooms.len());
                }
            }
        } else {
            render_data.objects.push(renderer::RenderObject {
                mesh_handle: renderer.upload_mesh(meshgen::generate_room_mesh(&database, 0)),
                material: renderer::Material::FlatLit,
            });
            room_meshes_uploaded.insert(0);
        }

        Ok(App {
            cx,
            renderer,
            render_data,
            room_meshes_uploaded,
            database,
        })
    }

    unsafe fn redraw(&mut self, delta: std::time::Duration) -> anyhow::Result<()> {
        self.renderer.draw(&mut self.cx, &self.render_data, delta)
    }
}

pub fn run(width: u32, height: u32) -> anyhow::Result<()> {
    let event_loop = EventLoop::new();
    let mut application = Some(unsafe { App::new(&event_loop, width, height)? });

    let mut last_time = std::time::Instant::now();
    let mut movement_keys_pressed = [false; 6];
    let mut sprint_key_pressed = false;
    let mut focused = false;
    let mut noclip = false;
    event_loop.run(move |event, _target, control_flow| {
        let Some(app) = application.as_mut() else { return };
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(new_size) => unsafe {
                    renderer::get_device().device_wait_idle().unwrap();
                    app.cx
                        .recreate_swapchain(new_size.width, new_size.height)
                        .unwrap();
                    app.renderer
                        .resize(&mut app.cx, new_size.width, new_size.height)
                        .unwrap();
                },
                WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state,
                            ..
                        },
                    ..
                } => match keycode {
                    winit::event::VirtualKeyCode::Tab
                        if state == winit::event::ElementState::Pressed =>
                    {
                        app.renderer.set_wireframe(!app.renderer.wireframe());
                    }
                    winit::event::VirtualKeyCode::N
                        if state == winit::event::ElementState::Pressed =>
                    {
                        noclip = !noclip;
                    }
                    winit::event::VirtualKeyCode::W => {
                        movement_keys_pressed[0] = state == winit::event::ElementState::Pressed
                    }
                    winit::event::VirtualKeyCode::S => {
                        movement_keys_pressed[1] = state == winit::event::ElementState::Pressed
                    }
                    winit::event::VirtualKeyCode::D => {
                        movement_keys_pressed[2] = state == winit::event::ElementState::Pressed
                    }
                    winit::event::VirtualKeyCode::A => {
                        movement_keys_pressed[3] = state == winit::event::ElementState::Pressed
                    }
                    winit::event::VirtualKeyCode::Space => {
                        movement_keys_pressed[4] = state == winit::event::ElementState::Pressed
                    }
                    winit::event::VirtualKeyCode::LShift => {
                        movement_keys_pressed[5] = state == winit::event::ElementState::Pressed
                    }
                    winit::event::VirtualKeyCode::LControl => {
                        sprint_key_pressed = state == winit::event::ElementState::Pressed
                    }

                    winit::event::VirtualKeyCode::Escape
                        if state == winit::event::ElementState::Pressed =>
                    {
                        focused = false;
                        app.cx.window.set_cursor_visible(true);
                        drop(app.cx.window.set_cursor_grab(CursorGrabMode::None));
                    }
                    _ => {}
                },
                WindowEvent::CursorMoved { .. } if focused => {
                    let PhysicalSize { width, height } = app.cx.window.inner_size();

                    drop(
                        app.cx
                            .window
                            .set_cursor_position(PhysicalPosition::new(width / 2, height / 2)),
                    );
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    ..
                } => {
                    focused = true;
                    app.cx.window.set_cursor_visible(false);
                    drop(app.cx.window.set_cursor_grab(CursorGrabMode::Confined));
                }
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            Event::DeviceEvent { event, .. } if focused => match event {
                winit::event::DeviceEvent::MouseMotion { delta } => {
                    const LOOK_SENSITIVITY: f32 = 0.001;
                    app.render_data.camera.look(
                        -delta.0 as f32 * LOOK_SENSITIVITY,
                        -delta.1 as f32 * LOOK_SENSITIVITY,
                    );
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                let player_pos = app.render_data.camera.position();
                let player_pos = ivec3(
                    player_pos.x.floor() as i32,
                    player_pos.y.floor() as i32,
                    (player_pos.z / CEILING_HEIGHT).floor() as i32,
                );
                if let Some(id) = app
                    .database
                    .map
                    .floor(player_pos.z)
                    .and_then(|floor| floor.cell(player_pos.truncate()))
                {
                    let current_room = &app.database.rooms[id];
                    let current_node = &app.database.nodes[current_room.node()];
                    // Load two children into the node tree
                    let nodes_to_load =
                        current_node.children.iter().copied().flat_map(|node_idx| {
                            std::iter::once(node_idx)
                                .chain(app.database.nodes[node_idx].children.iter().copied())
                        });
                    let rooms_to_load = nodes_to_load
                        .flat_map(|node| app.database.nodes[node].rooms.iter())
                        .filter(|room| !app.room_meshes_uploaded.contains(&room))
                        .collect::<Vec<_>>();
                    for &room_id in rooms_to_load {
                        app.render_data.objects.push(renderer::RenderObject {
                            mesh_handle: app
                                .renderer
                                .upload_mesh(meshgen::generate_room_mesh(&app.database, room_id)),
                            material: renderer::Material::FlatLit,
                        });
                        app.room_meshes_uploaded.insert(room_id);
                    }
                }
                app.cx.window.request_redraw();
            }
            Event::RedrawRequested(_) => unsafe {
                let now_time = std::time::Instant::now();
                let delta = now_time - last_time;
                let delta_s = delta.as_secs_f32();
                const SPEED: f32 = 5.;

                let mut movement = Vec3::ZERO;
                movement.z += if movement_keys_pressed[0] { 1. } else { 0. };
                movement.z -= if movement_keys_pressed[1] { 1. } else { 0. };
                movement.x -= if movement_keys_pressed[2] { 1. } else { 0. };
                movement.x += if movement_keys_pressed[3] { 1. } else { 0. };
                if noclip {
                    movement.y += if movement_keys_pressed[4] { 1. } else { 0. };
                    movement.y -= if movement_keys_pressed[5] { 1. } else { 0. };
                }
                if sprint_key_pressed {
                    movement *= 10.;
                }
                let movement = movement * SPEED * delta_s;
                let target_displacement = app
                    .render_data
                    .camera
                    .forward()
                    .truncate()
                    .extend(0.)
                    .normalize()
                    * movement.z
                    + app.render_data.camera.up() * movement.y
                    + app
                        .render_data
                        .camera
                        .right()
                        .truncate()
                        .extend(0.)
                        .normalize()
                        * movement.x;
                let starting_pos = app.render_data.camera.position();

                let current_pos = if noclip {
                    starting_pos + target_displacement
                } else {
                    move_and_slide(&app.database, starting_pos, target_displacement)
                };
                app.render_data.camera.set_position(current_pos);
                app.redraw(delta).unwrap();

                last_time = now_time;
            },
            Event::LoopDestroyed => unsafe {
                let mut app = application.take().unwrap();
                app.renderer.destroy(&mut app.cx).unwrap();
            },
            _ => {}
        }
    })
}
