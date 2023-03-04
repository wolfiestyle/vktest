use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use structopt::StructOpt;
use vkengine::gui::{egui, UiRenderer};
use vkengine::{CameraController, MeshRenderer, SkyboxRenderer, VulkanDevice, VulkanEngine, VulkanInstance, VulkanResult};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Fullscreen;

#[derive(StructOpt)]
struct Arguments {
    #[structopt(short, long, parse(from_os_str), help = "OBJ model file")]
    model: Option<PathBuf>,
    #[structopt(short, long, parse(from_os_str), help = "Texture for the model")]
    texture: Option<PathBuf>,
    #[structopt(short, long, parse(from_os_str), help = "Directory with cubemap images for the skybox")]
    skybox_dir: Option<PathBuf>,
}

fn main() -> VulkanResult<()> {
    let args = Arguments::from_args();

    let file = BufReader::new(File::open(args.model.unwrap_or_else(|| "data/model.obj".into())).unwrap());
    let model: obj::Obj<obj::TexturedVertex, u32> = obj::load_obj(file).unwrap();
    eprintln!("loaded {} vertices, {} indices", model.vertices.len(), model.indices.len());
    let image = image::open(args.texture.unwrap_or_else(|| "data/texture.png".into()))
        .unwrap()
        .into_rgba8();
    eprintln!("loaded image: {} x {}", image.width(), image.height());

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
    let win_size = winit::dpi::PhysicalSize::new(1024, 768);
    let window = winit::window::WindowBuilder::new()
        .with_title("vulkan test")
        .with_inner_size(win_size)
        .build(&event_loop)
        .unwrap();

    let vk_instance = VulkanInstance::new(&window, "vulkan test")?;
    let vk_device = VulkanDevice::new(&window, vk_instance, Default::default())?;
    let mut vk_app = VulkanEngine::new(vk_device, win_size.into())?;
    vk_app.camera.position = [2.0, 2.0, 2.0].into();
    vk_app.camera.look_at([0.0; 3]);

    let mut object = MeshRenderer::new(&vk_app, &model.vertices, &model.indices, image.dimensions(), image.as_raw())?;
    let mut skybox = SkyboxRenderer::new(&vk_app, skybox[0].dimensions(), &skybox_raw)?;

    let mut gui = UiRenderer::new(&event_loop, &vk_app)?;
    let mut show_gui = true;

    let mut prev_time = Instant::now();
    let mut prev_frame_count = 0;
    let mut fps = 0;
    let mut controller = CameraController::new();
    //TODO: compute these from current camera direction
    controller.yaw = 135.0;
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
                    eprintln!("{}", vk_app.device.get_memory_info());
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
                    vk_app.resize(size);
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
                        if ui.button("Exit").clicked() {
                            *control_flow = ControlFlow::Exit;
                        }
                    });
            });
            gui.event_output(&window);

            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            let draw_cmds = [
                object.render(&vk_app).unwrap(),
                skybox.render(&vk_app).unwrap(),
                gui.draw(&vk_app).unwrap(),
            ];

            vk_app.submit_draw_commands(draw_cmds).unwrap();

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
