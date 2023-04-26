use gltf_import::{GltfData, Material, Vertex};
use std::mem::ManuallyDrop;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use structopt::StructOpt;
use vkengine::gui::{egui, UiRenderer};
use vkengine::{Baker, CameraController, MeshRenderData, MeshRenderer, SkyboxRenderer, Texture, VkError, VulkanEngine, VulkanResult};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Fullscreen;

#[derive(StructOpt)]
struct Arguments {
    #[structopt(short, long, parse(from_os_str), help = "glTF model file")]
    model: Option<PathBuf>,
    #[structopt(short, long, parse(from_os_str), help = "File with the HDR panorama skybox")]
    skybox: Option<PathBuf>,
}

#[derive(Debug)]
pub struct MeshNode {
    renderer: MeshRenderer<Vertex, u32>,
    slices: Vec<MeshRenderData>,
    name: String,
    enabled: bool,
}

fn main() -> VulkanResult<()> {
    let args = Arguments::from_args();

    let gltf = GltfData::from_file(args.model.unwrap_or_else(|| "data/model.glb".into())).unwrap();
    for mesh in &gltf.meshes {
        println!(
            "loaded model: vertices={} indices={}, submeshes={}",
            mesh.vertices.len(),
            mesh.indices.len(),
            mesh.submeshes.len()
        );
    }

    let skybox_img = image::open(args.skybox.unwrap_or_else(|| "data/skybox.exr".into())).unwrap();

    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("vulkan test")
        .with_inner_size(PhysicalSize::new(1657, 1024))
        .build(&event_loop)
        .unwrap();

    let mut vk_app = VulkanEngine::new(&window, "vulkan test", Default::default())?;

    let mut resources = ManuallyDrop::new(vk_app.create_resources_for_model(&gltf).unwrap());

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
                            let desc_id = submesh.material.map(|mat| mat.0 + 1).unwrap_or_default();
                            MeshRenderData::from_gltf(submesh, material, resources.material_desc[desc_id])
                        })
                        .collect();
                    let renderer = MeshRenderer::new(&vk_app, &mesh.vertices, &mesh.indices, node.transform).unwrap();
                    let name = mesh.name.clone().unwrap_or_else(|| format!("Node {}", node.id.0));
                    MeshNode {
                        renderer,
                        slices,
                        name,
                        enabled: true,
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut cur_scene = 0;

    let camera_node = gltf
        .scenes
        .iter()
        .find_map(|scene| scene.nodes(&gltf).find(|&node| node.camera.is_some()));
    if let Some(node) = camera_node {
        let camera = &gltf[node.camera.unwrap()];
        vk_app.camera.set_transform(node.transform);
        if let gltf_import::Projection::Perspective { yfov, .. } = camera.projection {
            vk_app.camera.fov = yfov;
        }
    } else {
        vk_app.camera.position = [0.0, 1.0, 2.0].into();
        vk_app.camera.look_at([0.0; 3]);
    }
    let mut controller = CameraController::new(&vk_app.camera);

    let mut skybox = SkyboxRenderer::new(&vk_app)?;
    let skybox_equirect = Texture::from_dynamicimage(vk_app.device(), &skybox_img, false, false, Default::default())?;

    let mut gui = UiRenderer::new(&event_loop, &vk_app)?;
    let mut show_gui = true;

    let baker = Baker::new(&vk_app)?;
    let skybox_tex = baker.equirect_to_cubemap(&skybox_equirect, true)?;
    let irr_map = baker.generate_irradiance_map(&skybox_tex)?;
    let pref_map = baker.generate_prefilter_map(&skybox_tex)?;
    let brdf_lut = baker.generate_brdf_lut()?;

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
    let mut light = -vk_app.light;
    let mut light_directional = true;
    let mut fov = vk_app.camera.fov.to_degrees();

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
                        ui.checkbox(&mut vsync, "VSync");
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
                            ui.add(egui::Slider::new(&mut fov, 1.0..=120.0));
                        });
                        ui.add_space(10.0);
                        ui.label("Light:");
                        ui.horizontal(|ui| {
                            ui.label("X");
                            ui.add(egui::DragValue::new(&mut light.x).speed(0.1));
                            ui.label("Y");
                            ui.add(egui::DragValue::new(&mut light.y).speed(0.1));
                            ui.label("Z");
                            ui.add(egui::DragValue::new(&mut light.z).speed(0.1));
                        });
                        ui.checkbox(&mut light_directional, "Directional");
                        ui.add_space(10.0);
                        ui.collapsing(format!("{} Meshes", scenes[cur_scene].len()), |ui| {
                            egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                                for node in &mut scenes[cur_scene] {
                                    ui.add(egui::Checkbox::new(&mut node.enabled, &node.name));
                                }
                            });
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

            if light_directional {
                light.w = 0.0;
                vk_app.light = -light;
            } else {
                light.w = 1.0;
                vk_app.light = light;
            };

            vk_app.camera.fov = fov.to_radians();

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
                for (obj, draw_ret) in mesh_chunks.zip(ret_chunks) {
                    if obj[0].enabled {
                        scope.execute(|| {
                            draw_ret[0] = obj[0].renderer.render(&vk_app, &obj[0].slices, &irr_map, &pref_map, &brdf_lut);
                        });
                    }
                }
                scope.execute(|| skybox_cmds = skybox.render(&vk_app, &skybox_tex));
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
                cpu_time = vk_app.get_cpu_time();
                prev_time = cur_time;
                prev_frame_count = frame_count;
            }
        }
        Event::LoopDestroyed => unsafe {
            vk_app.device().device_wait_idle().unwrap();
            vk_app.device().dispose_of(ManuallyDrop::take(&mut resources));
        },
        _ => (),
    });
}
