use gltf_import::{Gltf, Material, Vertex};
use std::mem::ManuallyDrop;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use structopt::StructOpt;
use vkengine::gui::{egui, UiRenderer};
use vkengine::{CameraController, CubeData, MeshRenderSlice, MeshRenderer, SkyboxRenderer, VkError, VulkanEngine, VulkanResult};
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

#[derive(Debug)]
pub struct MeshNode {
    pub renderer: MeshRenderer<Vertex, u32>,
    pub slices: Vec<MeshRenderSlice>,
}

fn main() -> VulkanResult<()> {
    let args = Arguments::from_args();

    let gltf = Gltf::from_file(args.model.unwrap_or_else(|| "data/model.glb".into())).unwrap();
    for mesh in &gltf.meshes {
        println!(
            "loaded model: vertices={} indices={}, submeshes={}",
            mesh.vertices.len(),
            mesh.indices.len(),
            mesh.submeshes.len()
        );
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
    let cube_data = CubeData::try_from_iter(skybox.iter().map(|img| img.as_raw().as_slice())).unwrap();

    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("vulkan test")
        .with_inner_size(PhysicalSize::new(1657, 1024))
        .build(&event_loop)
        .unwrap();

    let mut vk_app = VulkanEngine::new(&window, "vulkan test", Default::default())?;

    let mut textures = ManuallyDrop::new(
        gltf.textures
            .iter()
            .map(|tex| vk_app.create_texture(tex, &gltf))
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
    );

    let mut scenes = gltf
        .scenes
        .iter()
        .map(|scene| {
            scene
                .nodes(&gltf)
                .filter_map(|node| node.mesh.map(|mesh_id| (&gltf[mesh_id], node)))
                .map(|(mesh, node)| {
                    let slices: Vec<_> = mesh
                        .submeshes
                        .iter()
                        .map(|submesh| {
                            let material = submesh.material.map(|mat| &gltf[mat]).unwrap_or(&Material::DEFAULT);
                            let color_tex = material.color_tex.map(|tex| textures[tex.id].descriptor());
                            let metal_rough_tex = material.metallic_roughness_tex.map(|tex| textures[tex.id].descriptor());
                            let normal_tex = material.normal_tex.map(|tex| textures[tex.id].descriptor());
                            let emiss_tex = material.emissive_tex.map(|tex| textures[tex.id].descriptor());
                            let occlusion_tex = material.occlusion_tex.map(|tex| textures[tex.id].descriptor());
                            MeshRenderSlice {
                                index_offset: submesh.index_offset,
                                index_count: submesh.index_count,
                                vertex_offset: submesh.vertex_offset,
                                base_color: material.base_color,
                                metallic: material.metallic,
                                roughness: material.roughness,
                                emissive: material.emissive,
                                normal_scale: material.normal_scale,
                                color_tex,
                                metal_rough_tex,
                                normal_tex,
                                emiss_tex,
                                occlusion_tex,
                            }
                        })
                        .collect();
                    let renderer = MeshRenderer::new(&vk_app, &mesh.vertices, &mesh.indices, &slices, node.transform).unwrap();
                    MeshNode { renderer, slices }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut mesh_enabled: Vec<Vec<_>> = scenes.iter().map(|meshes| vec![true; meshes.len()]).collect();
    let mut cur_scene = 0;

    vk_app.camera.position = [2.0, 2.0, 2.0].into();
    vk_app.camera.look_at([0.0; 3]);
    let mut controller = CameraController::new(&vk_app.camera);

    let mut skybox = SkyboxRenderer::new(&vk_app, skybox[0].dimensions(), cube_data)?;

    let mut gui = UiRenderer::new(&event_loop, &vk_app)?;
    let mut show_gui = true;

    let thread_pool = yastl::Pool::new(16);
    let mut draw_buffer = vec![];

    let mut prev_time = Instant::now();
    let mut prev_frame_count = 0;
    let mut fps = 0;
    let mut cpu_time = Default::default();
    let mut gpu_time = Default::default();

    let mut fullscreen = false;
    let mut msaa_samples = vk_app.get_msaa_samples();
    let mut vsync = vk_app.is_vsync_on();

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
                            virtual_keycode: Some(keycode),
                            ..
                        },
                    ..
                } => match keycode {
                    VirtualKeyCode::Escape => show_gui = !show_gui,
                    VirtualKeyCode::F1 => {
                        eprintln!("{}", vk_app.device().get_memory_info());
                    }
                    VirtualKeyCode::F11 => {
                        fullscreen = !fullscreen;
                        window.set_fullscreen(fullscreen.then_some(Fullscreen::Borderless(None)));
                    }
                    _ => controller.update_from_window_event(&event),
                },
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
                        ui.toggle_value(&mut vsync, "VSync");
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
                        mesh.renderer.rebuild_pipeline(&vk_app).unwrap();
                    }
                }

                skybox.rebuild_pipeline(&vk_app).unwrap();
                gui.rebuild_pipeline(&vk_app).unwrap();
            }

            if vsync != vk_app.is_vsync_on() {
                vk_app.set_vsync(vsync).unwrap();
            }

            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            vk_app.update();

            let objects = &mut scenes[cur_scene];
            draw_buffer.resize_with(objects.len(), || VkError::UnfinishedJob.into());
            let mut skybox_cmds = VkError::UnfinishedJob.into();
            let mut gui_cmds = VkError::UnfinishedJob.into();
            thread_pool.scoped(|scope| {
                let mesh_chunks = objects.chunks_mut(1);
                let ret_chunks = draw_buffer.chunks_mut(1);
                for (i, (obj, draw_ret)) in mesh_chunks.zip(ret_chunks).enumerate() {
                    if mesh_enabled[cur_scene][i] {
                        scope.execute(|| {
                            draw_ret[0] = obj[0].renderer.render(&vk_app, &obj[0].slices);
                        });
                    }
                }
                scope.execute(|| skybox_cmds = skybox.render(&vk_app));
                scope.execute(|| gui_cmds = gui.draw(&vk_app));
            });

            let draw_cmds = draw_buffer.drain(..).chain([skybox_cmds, gui_cmds]).filter_map(|res| res.ok());

            vk_app.submit_draw_commands(draw_cmds).unwrap();

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
        Event::LoopDestroyed => unsafe {
            vk_app.device().device_wait_idle().unwrap();
            vk_app.device().dispose_of(ManuallyDrop::take(&mut textures));
        },
        _ => (),
    });
}
