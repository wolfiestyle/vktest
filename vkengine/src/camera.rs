use glam::{Affine3A, EulerRot, Mat4, Quat, Vec3};
#[cfg(feature = "winit")]
use winit::event::{DeviceEvent, WindowEvent};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    pub rotation: Quat,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn look_at(&mut self, center: impl Into<Vec3>) {
        let dir = center.into() - self.position;
        if let Some(dir) = dir.try_normalize() {
            let pitch = (dir.y).asin();
            let yaw = (-dir.x).atan2(-dir.z);
            self.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
        }
    }

    pub fn fly_forward(&mut self, dist: f32) {
        self.position += self.rotation * Vec3::Z * dist;
    }

    pub fn fly_up(&mut self, dist: f32) {
        self.position += Vec3::Y * dist;
    }

    pub fn walk_forward(&mut self, dist: f32, ground: impl Into<Vec3>) {
        self.position += (self.rotation * Vec3::X).cross(ground.into()).normalize() * dist;
    }

    pub fn walk_right(&mut self, dist: f32) {
        self.position += self.rotation * Vec3::X * dist;
    }

    pub fn set_rotation(&mut self, pitch: f32, yaw: f32, roll: f32) {
        self.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);
    }

    pub(crate) fn get_view_transform(&self) -> Affine3A {
        Affine3A::from_quat(self.rotation.conjugate()) * Affine3A::from_translation(-self.position)
    }

    pub(crate) fn get_view_rotation(&self) -> Affine3A {
        Affine3A::from_quat(self.rotation.conjugate())
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
            rotation: Quat::IDENTITY,
            fov: 60.0,
            near: 0.05,
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
    pub roll: f32,
}

impl CameraController {
    pub fn new(camera: &Camera) -> Self {
        let (y, x, z) = camera.rotation.to_euler(EulerRot::YXZ);
        Self {
            speed: 1.0,
            flying: false,
            forward_speed: 0.0,
            right_speed: 0.0,
            up_speed: 0.0,
            mouse_look: false,
            sensitivity: 0.1,
            yaw: y.to_degrees(),
            pitch: x.to_degrees(),
            roll: z.to_degrees(),
        }
    }

    pub fn update_camera(&self, camera: &mut Camera, dt: f32) {
        if self.forward_speed != 0.0 {
            if self.flying {
                camera.fly_forward(self.forward_speed * dt);
            } else {
                camera.walk_forward(self.forward_speed * dt, Vec3::Y);
            }
        }
        if self.right_speed != 0.0 {
            camera.walk_right(self.right_speed * dt);
        }
        if self.up_speed != 0.0 {
            camera.fly_up(self.up_speed * dt);
        }
        if self.mouse_look {
            camera.set_rotation(self.pitch.to_radians(), self.yaw.to_radians(), self.roll.to_radians());
        }
    }

    #[cfg(feature = "winit")]
    #[rustfmt::skip]
    pub fn update_from_window_event(&mut self, event: &WindowEvent) {
        use winit::event::{ElementState::*, KeyboardInput, MouseButton::*, VirtualKeyCode as Key};

        match event {
            WindowEvent::KeyboardInput { input: KeyboardInput { state, virtual_keycode: Some(keycode), .. }, .. } => match (state, keycode) {
                (Pressed, Key::W) => {
                    self.forward_speed = -self.speed;
                }
                (Pressed, Key::S) => {
                    self.forward_speed = self.speed;
                }
                (Released, Key::W | Key::S) => {
                    self.forward_speed = 0.0;
                }
                (Pressed, Key::A) => {
                    self.right_speed = -self.speed;
                }
                (Pressed, Key::D) => {
                    self.right_speed = self.speed;
                }
                (Released, Key::A | Key::D) => {
                    self.right_speed = 0.0;
                }
                (Pressed, Key::Space) => {
                    self.up_speed = self.speed;
                }
                (Pressed, Key::C) => {
                    self.up_speed = -self.speed;
                }
                (Released, Key::Space | Key::C) => {
                    self.up_speed = 0.0;
                }
                (Released, Key::F) => {
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
                let yaw = self.yaw - dx as f32 * self.sensitivity;
                let pitch = self.pitch - dy as f32 * self.sensitivity;
                self.yaw = yaw % 360.0;
                self.pitch = pitch.clamp(-89.0, 89.0);
            }
            _ => (),
        }
    }
}
