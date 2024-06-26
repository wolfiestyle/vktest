use glam::{Affine3A, EulerRot, Mat4, Quat, Vec3};
#[cfg(feature = "winit")]
use winit::event::{DeviceEvent, WindowEvent};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    pub rotation: Quat,
    pub fov: f32,
    pub near: f32,
    pub far: Option<f32>,
}

impl Camera {
    pub fn from_gltf(camera: &gltf_import::Camera, node: &gltf_import::Node) -> Self {
        let mut this = Self::default();
        this.set_transform(node.transform);
        if let gltf_import::Projection::Perspective { yfov, znear, zfar, .. } = camera.projection {
            this.fov = yfov;
            this.near = znear;
            this.far = zfar;
        }
        this
    }

    pub fn look_at(&mut self, center: Vec3) {
        let dir = center - self.position;
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

    pub fn walk_forward(&mut self, dist: f32, ground: Vec3) {
        self.position += (self.rotation * Vec3::X).cross(ground).normalize() * dist;
    }

    pub fn walk_right(&mut self, dist: f32) {
        self.position += self.rotation * Vec3::X * dist;
    }

    pub fn set_rotation(&mut self, pitch: f32, yaw: f32, roll: f32) {
        self.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);
    }

    pub fn set_transform(&mut self, affine: Affine3A) {
        let (_, rotation, translation) = affine.to_scale_rotation_translation();
        self.position = translation;
        self.rotation = rotation;
    }

    pub(crate) fn get_view_transform(&self) -> Affine3A {
        Affine3A::from_quat(self.rotation.conjugate()) * Affine3A::from_translation(-self.position)
    }

    pub(crate) fn get_view_rotation(&self) -> Affine3A {
        Affine3A::from_quat(self.rotation.conjugate())
    }

    pub(crate) fn get_projection(&self, aspect: f32, reverse_depth: bool) -> Mat4 {
        if reverse_depth {
            Mat4::perspective_infinite_reverse_rh(self.fov, aspect, self.near)
        } else if let Some(far) = self.far {
            Mat4::perspective_rh(self.fov, aspect, self.near, far)
        } else {
            Mat4::perspective_infinite_rh(self.fov, aspect, self.near)
        }
    }
}

impl Default for Camera {
    #[inline]
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            fov: 60.0_f32.to_radians(),
            near: 0.05,
            far: None,
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
        camera.set_rotation(self.pitch.to_radians(), self.yaw.to_radians(), self.roll.to_radians());
    }

    #[cfg(feature = "winit")]
    #[rustfmt::skip]
    pub fn update_from_window_event(&mut self, event: &WindowEvent) {
        use winit::event::{ElementState::*, KeyEvent, MouseButton::*};
        use winit::keyboard::{PhysicalKey, KeyCode::*};

        match event {
            WindowEvent::KeyboardInput { event: KeyEvent { state, physical_key: PhysicalKey::Code(keycode), .. }, .. } => match (state, keycode) {
                (Pressed, KeyW) => {
                    self.forward_speed = -self.speed;
                }
                (Pressed, KeyS) => {
                    self.forward_speed = self.speed;
                }
                (Released, KeyW | KeyS) => {
                    self.forward_speed = 0.0;
                }
                (Pressed, KeyA) => {
                    self.right_speed = -self.speed;
                }
                (Pressed, KeyD) => {
                    self.right_speed = self.speed;
                }
                (Released, KeyA | KeyD) => {
                    self.right_speed = 0.0;
                }
                (Pressed, Space) => {
                    self.up_speed = self.speed;
                }
                (Pressed, KeyC) => {
                    self.up_speed = -self.speed;
                }
                (Released, Space | KeyC) => {
                    self.up_speed = 0.0;
                }
                (Released, KeyF) => {
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
