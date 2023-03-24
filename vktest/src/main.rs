use easy_gltf::model::Mode;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use structopt::StructOpt;
use vkengine::gui::{egui, UiRenderer};
use vkengine::{CameraController, MeshRenderer, SkyboxRenderer, VkError, VulkanEngine, VulkanResult};
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

    let gltf_scenes = easy_gltf::load(args.model.unwrap_or_else(|| "data/model.glb".into())).unwrap();
    for scene in &gltf_scenes {
        for model in &scene.models {
            println!(
                "loaded model: vertices={} indices={:?}, mode={:?}",
                model.vertices().len(),
                model.indices().map(Vec::len),
                model.mode()
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
        .with_inner_size(PhysicalSize::new(1657, 1024))
        .build(&event_loop)
        .unwrap();

    let mut vk_app = VulkanEngine::new(&window, "vulkan test", Default::default())?;

    let mut scenes = gltf_scenes
        .iter()
        .map(|scene| {
            scene
                .models
                .iter()
                .map(|model| {
                    //FIXME: possible duplicate texture creation
                    let color_tex = model
                        .material()
                        .pbr
                        .base_color_texture
                        .as_ref()
                        .map(|image| vk_app.create_texture(image.width(), image.height(), image.as_raw()).unwrap());
                    let indices = model.indices().map(|vec| vec.as_slice());
                    let color = model.material().pbr.base_color_factor.into();
                    MeshRenderer::new(&vk_app, model.vertices(), indices, color, color_tex)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    //FIXME: disabling meshes not made of triangles. implement other modes or properly skip them
    let mut mesh_enabled: Vec<Vec<_>> = gltf_scenes
        .iter()
        .map(|scene| scene.models.iter().map(|model| model.mode() == Mode::Triangles).collect())
        .collect();
    let mut cur_scene = 0;

    vk_app.camera.position = [2.0, 2.0, -2.0].into();
    vk_app.camera.look_at([0.0; 3]);
    let mut controller = CameraController::new(&vk_app.camera);

    let mut skybox = SkyboxRenderer::new(&vk_app, skybox[0].dimensions(), &skybox_raw)?;

    let mut gui = UiRenderer::new(&event_loop, &vk_app)?;
    let mut show_gui = true;

    let thread_pool = yastl::Pool::new(4);

    let mut prev_time = Instant::now();
    let mut prev_frame_count = 0;
    let mut fps = 0;
    let mut cpu_time = Default::default();
    let mut gpu_time = Default::default();

    let mut fullscreen = false;
    let mut msaa_samples = vk_app.get_msaa_samples();

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
                    .frame(egui::Frame::default().fill(egui::Color32::from_black_alpha(225)))
                    .resizable(false)
                    .show_animated(ctx, show_gui, |ui| {
                        ui.heading("VKtest 3D engine");
                        ui.label(format!("{fps} fps"));
                        ui.label(format!("cpu time: {:?}", cpu_time));
                        ui.label(format!("gpu time: {:?}", gpu_time));
                        ui.add_space(10.0);
                        egui::ComboBox::from_label("MSAA")
                            .selected_text(format!("{msaa_samples}x"))
                            .show_ui(ui, |ui| {
                                let supp = vk_app.device().dev_info.msaa_support.as_raw();
                                for i in 0..=6 {
                                    let val = 1 << i;
                                    if val & supp != 0 {
                                        ui.selectable_value(&mut msaa_samples, val, format!("{val}x"));
                                    }
                                }
                            });
                        ui.add_space(10.0);
                        egui::ComboBox::from_label("Scene")
                            .selected_text(format!("Scene {cur_scene}"))
                            .show_ui(ui, |ui| {
                                for i in 0..scenes.len() {
                                    ui.selectable_value(&mut cur_scene, i, format!("Scene {i}"));
                                }
                            });
                        ui.label("Camera:");
                        ui.horizontal(|ui| {
                            ui.label("X");
                            ui.add(egui::DragValue::new(&mut vk_app.camera.position.x).speed(0.1));
                            ui.label("Y");
                            ui.add(egui::DragValue::new(&mut vk_app.camera.position.y).speed(0.1));
                            ui.label("Z");
                            ui.add(egui::DragValue::new(&mut vk_app.camera.position.z).speed(0.1));
                        });
                        ui.checkbox(&mut controller.flying, "Flying");
                        ui.horizontal(|ui| {
                            ui.label("fov: ");
                            ui.add(egui::Slider::new(&mut vk_app.camera.fov, 10.0..=120.0));
                        });
                        ui.add_space(10.0);
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
                        ui.collapsing(format!("{} Meshes", mesh_enabled[cur_scene].len()), |ui| {
                            for (i, enable) in mesh_enabled[cur_scene].iter_mut().enumerate() {
                                ui.add(egui::Checkbox::new(enable, format!("Mesh {i}")));
                            }
                        });
                        ui.add_space(10.0);
                        if ui.button("Exit").clicked() {
                            *control_flow = ControlFlow::Exit;
                        }
                    });
            });
            gui.event_output(&window);

            if msaa_samples != vk_app.get_msaa_samples() {
                vk_app.set_msaa_samples(msaa_samples).unwrap();

                for scene in &mut scenes {
                    for mesh in scene {
                        mesh.rebuild_pipeline(&vk_app).unwrap();
                    }
                }
                skybox.rebuild_pipeline(&vk_app).unwrap();
                gui.rebuild_pipeline(&vk_app).unwrap();
            }

            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            vk_app.update();

            let objects = &mut scenes[cur_scene];
            let mut draw_cmds = Vec::with_capacity(objects.len());
            let mut skybox_cmds = VkError::UnfinishedJob.into();
            let mut gui_cmds = VkError::UnfinishedJob.into();
            thread_pool.scoped(|scope| {
                scope.execute(|| {
                    draw_cmds.extend(
                        objects
                            .iter_mut()
                            .enumerate()
                            .filter(|&(i, _)| mesh_enabled[cur_scene][i])
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
            let dt = cur_time - prev_time;
            if dt >= Duration::from_millis(500) {
                let frame_count = vk_app.get_current_frame();
                fps = (frame_count - prev_frame_count) * 1000 / dt.as_millis() as u64;
                gpu_time = vk_app.get_gpu_time();
                cpu_time = vk_app.get_frame_time().saturating_sub(gpu_time);
                prev_time = cur_time;
                prev_frame_count = frame_count;
            }
        }
        _ => (),
    });
}
