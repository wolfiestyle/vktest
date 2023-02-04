use std::time::{Duration, Instant};
use vktest::vulkan::VulkanApp;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("vulkan test")
        .with_inner_size(winit::dpi::LogicalSize::new(1024, 768))
        .build(&event_loop)
        .unwrap();

    let mut vk_app = VulkanApp::new(&window).unwrap();
    let mut prev_time = Instant::now();
    let mut frame_count = 0;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event:
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                },
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            vk_app.draw_frame().unwrap();

            let cur_time = Instant::now();
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
