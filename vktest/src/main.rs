use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use structopt::StructOpt;
use vkengine::{VulkanDevice, VulkanEngine, VulkanInstance, VulkanResult};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

#[derive(StructOpt)]
struct Arguments {
    #[structopt(short, long, parse(from_os_str), help = "OBJ model file")]
    model: Option<PathBuf>,
    #[structopt(short, long, parse(from_os_str), help = "Texture for the model")]
    texture: Option<PathBuf>,
}

fn main() -> VulkanResult<()> {
    let args = Arguments::from_args();

    let file = BufReader::new(File::open(args.model.unwrap_or_else(|| "model.obj".into())).unwrap());
    let model = obj::load_obj(file).unwrap();
    eprintln!("loaded {} vertices, {} indices", model.vertices.len(), model.indices.len());
    let image = image::open(args.texture.unwrap_or_else(|| "texture.png".into()))
        .unwrap()
        .into_rgba8();
    eprintln!("loaded image: {} x {}", image.width(), image.height());

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
        image.width(),
        image.height(),
        image.as_raw(),
    )?;
    let mut prev_time = Instant::now();
    let mut frame_count = 0u32;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
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
            WindowEvent::Resized(size) => {
                vk_app.resize(size.into());
            }
            _ => (),
        },
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            vk_app.draw_frame().unwrap();

            let cur_time = vk_app.get_frame_time();
            if cur_time - prev_time > Duration::from_secs(1) {
                println!("{frame_count} fps");
                prev_time = cur_time;
                frame_count = 0;
            } else {
                frame_count += 1;
            }
        }
        _ => (),
    });
}
