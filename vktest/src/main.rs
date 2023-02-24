use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use structopt::StructOpt;
use vkengine::{CameraController, VulkanDevice, VulkanEngine, VulkanInstance, VulkanResult};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Fullscreen;

#[derive(StructOpt)]
struct Arguments {
    #[structopt(short, long, parse(from_os_str), help = "OBJ model file")]
    model: Option<PathBuf>,
    #[structopt(short, long, parse(from_os_str), help = "Texture for the model")]
    texture: Option<PathBuf>,
}

fn main() -> VulkanResult<()> {
    let args = Arguments::from_args();

    let file = BufReader::new(File::open(args.model.unwrap_or_else(|| "data/model.obj".into())).unwrap());
    let model = obj::load_obj(file).unwrap();
    eprintln!("loaded {} vertices, {} indices", model.vertices.len(), model.indices.len());
    let image = image::open(args.texture.unwrap_or_else(|| "data/texture.png".into()))
        .unwrap()
        .into_rgba8();
    eprintln!("loaded image: {} x {}", image.width(), image.height());

    let skybox = ["posx", "negx", "posy", "negy", "posz", "negz"].map(|side| {
        let filename = format!("data/skybox/{side}.jpg");
        eprintln!("loading {filename}");
        image::open(filename).unwrap().into_rgba8()
    });
    let skybox_raw: Vec<_> = skybox.iter().map(|img| img.as_raw().as_slice()).collect();

    let event_loop = EventLoop::new();
    let win_size = winit::dpi::PhysicalSize::new(1024, 768);
    let window = winit::window::WindowBuilder::new()
        .with_title("vulkan test")
        .with_inner_size(win_size)
        .build(&event_loop)
        .unwrap();

    let vk_instance = VulkanInstance::new(&window, "vulkan test")?;
    let vk_device = VulkanDevice::new(&window, vk_instance, Default::default())?;
    let mut vk_app = VulkanEngine::new(
        vk_device,
        win_size.into(),
        &model.vertices,
        &model.indices,
        image.dimensions(),
        image.as_raw(),
        skybox[0].dimensions(),
        &skybox_raw,
    )?;
    vk_app.camera.position = [2.0, 2.0, 2.0].into();
    vk_app.camera.look_at([0.0; 3]);

    let mut prev_time = Instant::now();
    let mut frame_count = 0u32;
    let mut controller = CameraController::new();
    //TODO: compute these from current camera direction
    controller.yaw = 135.0;
    controller.pitch = 35.0;

    let mut fullscreen = false;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
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
                *control_flow = ControlFlow::Exit;
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
        },
        Event::DeviceEvent { event, .. } => controller.update_from_device_event(&event),
        Event::MainEventsCleared => {
            let dt = (vk_app.get_frame_time().as_micros() as f64 / 1e6) as f32;
            controller.update_camera(&mut vk_app.camera, dt);

            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            vk_app.draw_frame().unwrap();

            let cur_time = vk_app.get_frame_timestamp();
            if cur_time - prev_time > Duration::from_secs(1) {
                println!("{frame_count} fps, frame time {:?}", vk_app.get_frame_time());
                prev_time = cur_time;
                frame_count = 0;
            } else {
                frame_count += 1;
            }
        }
        _ => (),
    });
}
