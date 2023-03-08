use std::path::PathBuf;
use std::time::{Duration, Instant};
use structopt::StructOpt;
use vkengine::gui::{egui, UiRenderer};
use vkengine::{CameraController, MeshRenderer, SkyboxRenderer, VertexTemp, VkError, VulkanEngine, VulkanResult};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Fullscreen;

#[derive(StructOpt)]
struct Arguments {
    #[structopt(short, long, parse(from_os_str), help = "glTF model file")]
    model: Option<PathBuf>,
    #[structopt(short, long, parse(from_os_str), help = "Directory with cubemap images for the skybox")]
    skybox_dir: Option<PathBuf>,
}

fn main() -> VulkanResult<()> {
    let args = Arguments::from_args();

    let scenes = easy_gltf::load(args.model.unwrap_or_else(|| "data/model.glb".into())).unwrap();
    for scene in &scenes {
        for model in &scene.models {
            println!(
                "loaded model: vertices={} indices={}",
                model.vertices().len(),
                model.indices().unwrap().len()
            );
        }
    }

    let skybox_dir = args.skybox_dir.unwrap_or_else(|| "data/skybox".into());
    let skybox = std::thread::scope(|scope| {
        let dir_ref = &skybox_dir;
        ["posx", "negx", "posy", "negy", "posz", "negz"]
            .map(|side| {
                scope.spawn(move || {
                    let filename = format!("{side}.jpg"); //FIXME: detect format
                    let path = dir_ref.join(filename);
                    eprintln!("loading {path:?}");
                    image::open(path).unwrap().into_rgba8()
                })
            })
            .map(|jh| jh.join().unwrap())
    });
    let skybox_raw = skybox
        .iter()
        .map(|img| img.as_raw().as_slice())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("vulkan test")
        .with_inner_size(PhysicalSize::new(1024, 768))
        .build(&event_loop)
        .unwrap();

    let mut vk_app = VulkanEngine::new(&window, "vulkan test", Default::default())?;
    vk_app.camera.position = [2.0, 2.0, 2.0].into();
    vk_app.camera.look_at([0.0; 3]);

    let mut objects = scenes
        .iter()
        .map(|scene| {
            scene.models.iter().map(|model| {
                let texture = model.material().pbr.base_color_texture.clone().unwrap();
                //HACK: vertices are returned on a non-Copy, non repr-C struct, so can't feed it directly
                let (p, verts, s) = unsafe { model.vertices().align_to::<VertexTemp>() };
                assert!(p.len() == 0 && s.len() == 0);
                //also indices are returned in usize for some reason
                let indices: Vec<_> = model.indices().unwrap().iter().map(|&idx| idx as u32).collect();
                MeshRenderer::new(&vk_app, verts, &indices, texture.dimensions(), texture.as_raw())
            })
        })
        .flatten()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let mut mesh_enabled = vec![true; objects.len()];

    let mut skybox = SkyboxRenderer::new(&vk_app, skybox[0].dimensions(), &skybox_raw)?;

    let mut gui = UiRenderer::new(&event_loop, &vk_app)?;
    let mut show_gui = true;

    let thread_pool = yastl::Pool::new(4);

    let mut prev_time = Instant::now();
    let mut prev_frame_count = 0;
    let mut fps = 0;
    let mut controller = CameraController::new();
    //TODO: compute these from current camera direction
    controller.yaw = 225.0;
    controller.pitch = 35.0;

    let mut fullscreen = false;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => {
            if gui.event_input(&event).consumed {
                return;
            }

            match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => {
                    show_gui = !show_gui;
                }
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::F1),
                            ..
                        },
                    ..
                } => {
                    eprintln!("{}", vk_app.device().get_memory_info());
                }
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::F11),
                            ..
                        },
                    ..
                } => {
                    fullscreen = !fullscreen;
                    window.set_fullscreen(fullscreen.then_some(Fullscreen::Borderless(None)));
                }
                WindowEvent::Resized(size) => {
                    vk_app.resize(size.width, size.height);
                }
                _ => controller.update_from_window_event(&event),
            }
        }
        Event::DeviceEvent { event, .. } => controller.update_from_device_event(&event),
        Event::MainEventsCleared => {
            let dt = (vk_app.get_frame_time().as_micros() as f64 / 1e6) as f32;
            controller.update_camera(&mut vk_app.camera, dt);

            gui.run(&window, |ctx| {
                egui::panel::SidePanel::left("main")
                    .frame(egui::Frame::default().fill(egui::Color32::from_black_alpha(192)))
                    .resizable(false)
                    .show_animated(ctx, show_gui, |ui| {
                        ui.heading("VKtest 3D engine");
                        ui.label(format!("{fps} fps"));
                        ui.label(format!("frame {}", vk_app.get_frame_count()));
                        ui.add_space(10.0);
                        ui.checkbox(&mut controller.flying, "Flying");
                        ui.horizontal(|ui| {
                            ui.label("fov: ");
                            ui.add(egui::Slider::new(&mut vk_app.camera.fov, 10.0..=120.0));
                        });
                        ui.label("Sunlight:");
                        ui.horizontal(|ui| {
                            ui.label("X");
                            ui.add(egui::DragValue::new(&mut vk_app.sunlight.x).speed(0.1));
                            ui.label("Y");
                            ui.add(egui::DragValue::new(&mut vk_app.sunlight.y).speed(0.1));
                            ui.label("Z");
                            ui.add(egui::DragValue::new(&mut vk_app.sunlight.z).speed(0.1));
                        });
                        if ui.button("Normalize").clicked() {
                            if let Some(v) = vk_app.sunlight.try_normalize() {
                                vk_app.sunlight = v;
                            }
                        }
                        ui.add_space(10.0);
                        for (i, enable) in mesh_enabled.iter_mut().enumerate() {
                            ui.add(egui::Checkbox::new(enable, format!("Mesh {i}")));
                        }
                        ui.add_space(10.0);
                        if ui.button("Exit").clicked() {
                            *control_flow = ControlFlow::Exit;
                        }
                    });
            });
            gui.event_output(&window);

            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            let mut draw_cmds = Vec::with_capacity(objects.len());
            let mut skybox_cmds = VkError::UnfinishedJob.into();
            let mut gui_cmds = VkError::UnfinishedJob.into();
            thread_pool.scoped(|scope| {
                scope.execute(|| {
                    draw_cmds.extend(
                        objects
                            .iter_mut()
                            .enumerate()
                            .filter(|&(i, _)| mesh_enabled[i])
                            .map(|(_, obj)| obj.render(&vk_app)),
                    )
                });
                scope.execute(|| skybox_cmds = skybox.render(&vk_app));
                scope.execute(|| gui_cmds = gui.draw(&vk_app));
            });

            draw_cmds.push(skybox_cmds);
            draw_cmds.push(gui_cmds);

            vk_app.submit_draw_commands(draw_cmds.into_iter().map(|res| res.unwrap())).unwrap();

            let cur_time = vk_app.get_frame_timestamp();
            if cur_time - prev_time > Duration::from_secs(1) {
                let frame_count = vk_app.get_frame_count();
                fps = frame_count - prev_frame_count;
                println!("{fps} fps, frame time {:?}", vk_app.get_frame_time());
                prev_time = cur_time;
                prev_frame_count = frame_count;
            }
        }
        _ => (),
    });
}
