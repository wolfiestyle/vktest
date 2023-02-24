use glam::{Affine3A, EulerRot, Mat4, Quat, Vec3};
#[cfg(feature = "winit")]
use winit::event::{DeviceEvent, WindowEvent};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    pub direction: Vec3,
    pub up: Vec3,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(position: impl Into<Vec3>, direction: impl Into<Vec3>, up: impl Into<Vec3>) -> Self {
        Self {
            position: position.into(),
            direction: direction.into(),
            up: up.into(),
            ..Default::default()
        }
    }

    pub fn look_at(&mut self, center: impl Into<Vec3>) {
        let dir = center.into() - self.position;
        if dir != Vec3::ZERO {
            self.direction = dir;
        }
    }

    pub fn fly_forward(&mut self, dist: f32) {
        self.position += self.direction.normalize() * dist;
    }

    pub fn fly_up(&mut self, dist: f32) {
        self.position += self.up.normalize() * dist;
    }

    pub fn walk_forward(&mut self, dist: f32, ground: impl Into<Vec3>) {
        self.position += self.up.cross(self.direction).cross(ground.into()).normalize() * dist;
    }

    pub fn walk_right(&mut self, dist: f32) {
        self.position += self.direction.cross(self.up).normalize() * dist;
    }

    pub fn set_rotation(&mut self, pitch: f32, yaw: f32) {
        let quat = Quat::from_euler(EulerRot::ZXY, yaw, -pitch, 0.0);
        self.direction = quat * Vec3::Y;
    }

    pub(crate) fn get_view_transform(&self) -> Affine3A {
        Affine3A::look_to_rh(self.position, self.direction, self.up)
    }

    pub(crate) fn get_projection(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov.to_radians(), aspect, self.near, self.far)
    }
}

impl Default for Camera {
    #[inline]
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            direction: Vec3::Y,
            up: Vec3::NEG_Z,
            fov: 45.0,
            near: 0.1,
            far: 1000.0,
        }
    }
}

//FIXME: needs to account for simultaneous opposite movements and order of actions
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraController {
    pub speed: f32,
    pub flying: bool,
    pub forward_speed: f32,
    pub right_speed: f32,
    pub up_speed: f32,
    pub mouse_look: bool,
    pub sensitivity: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            speed: 1.0,
            flying: false,
            forward_speed: 0.0,
            right_speed: 0.0,
            up_speed: 0.0,
            mouse_look: false,
            sensitivity: 0.1,
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera, dt: f32) {
        if self.forward_speed != 0.0 {
            if self.flying {
                camera.fly_forward(self.forward_speed * dt);
            } else {
                camera.walk_forward(self.forward_speed * dt, camera.up);
            }
        }
        if self.right_speed != 0.0 {
            camera.walk_right(self.right_speed * dt);
        }
        if self.up_speed != 0.0 {
            camera.fly_up(self.up_speed * dt);
        }
        if self.mouse_look {
            camera.set_rotation(self.pitch.to_radians(), self.yaw.to_radians());
        }
    }

    #[cfg(feature = "winit")]
    #[rustfmt::skip]
    pub fn update_from_window_event(&mut self, event: &WindowEvent) {
        use winit::event::{ElementState::*, KeyboardInput, MouseButton::*, VirtualKeyCode as Key};

        match event {
            WindowEvent::KeyboardInput { input, .. } => match input {
                KeyboardInput { state: Pressed, virtual_keycode: Some(Key::W), .. } => {
                    self.forward_speed = self.speed;
                }
                KeyboardInput { state: Pressed, virtual_keycode: Some(Key::S), .. } => {
                    self.forward_speed = -self.speed;
                }
                KeyboardInput { state: Released, virtual_keycode: Some(Key::W | Key::S), .. } => {
                    self.forward_speed = 0.0;
                }
                KeyboardInput { state: Pressed, virtual_keycode: Some(Key::A), .. } => {
                    self.right_speed = -self.speed;
                }
                KeyboardInput { state: Pressed, virtual_keycode: Some(Key::D), .. } => {
                    self.right_speed = self.speed;
                }
                KeyboardInput { state: Released, virtual_keycode: Some(Key::A | Key::D), .. } => {
                    self.right_speed = 0.0;
                }
                KeyboardInput { state: Pressed, virtual_keycode: Some(Key::Space), .. } => {
                    self.up_speed = -self.speed;
                }
                KeyboardInput { state: Pressed, virtual_keycode: Some(Key::C), .. } => {
                    self.up_speed = self.speed;
                }
                KeyboardInput { state: Released, virtual_keycode: Some(Key::Space | Key::C), .. } => {
                    self.up_speed = 0.0;
                }
                KeyboardInput { state: Released, virtual_keycode: Some(Key::F), .. } => {
                    self.flying = !self.flying;
                }
                _ => (),
            }
            WindowEvent::MouseInput { state: Pressed, button: Left, .. } => {
                self.mouse_look = true;
            }
            WindowEvent::MouseInput { state: Released, button: Left, .. } => {
                self.mouse_look = false;
            }
            _ => (),
        }
    }

    #[cfg(feature = "winit")]
    pub fn update_from_device_event(&mut self, event: &DeviceEvent) {
        match event {
            &DeviceEvent::MouseMotion { delta: (dx, dy) } if self.mouse_look => {
                let yaw = self.yaw + dx as f32 * self.sensitivity;
                let pitch = self.pitch + dy as f32 * self.sensitivity;
                self.yaw = yaw % 360.0;
                self.pitch = pitch.clamp(-89.0, 89.0);
            }
            _ => (),
        }
    }
}
