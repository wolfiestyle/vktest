use glam::{Affine3A, EulerRot, Mat4, Quat, Vec3, camera::rh::proj::vulkan};
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
            vulkan::perspective_infinite_reverse(self.fov, aspect, self.near)
        } else if let Some(far) = self.far {
            vulkan::perspective(self.fov, aspect, self.near, far)
        } else {
            vulkan::perspective_infinite(self.fov, aspect, self.near)
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

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn assert_depth(projection: Mat4, distance: f32, expected: f32) {
        let depth = projection.project_point3(Vec3::NEG_Z * distance).z;
        assert!(
            (depth - expected).abs() < EPSILON,
            "depth at distance {distance}: expected {expected}, got {depth}"
        );
    }

    #[test]
    fn view_transform_places_the_eye_at_the_origin_and_target_in_front() {
        let mut camera = Camera {
            position: Vec3::new(3.0, -2.0, 5.0),
            ..Camera::default()
        };
        let offset = Vec3::new(2.0, 3.0, -4.0);
        let target = camera.position + offset;
        camera.look_at(target);

        let view = camera.get_view_transform();
        assert!(view.transform_point3(camera.position).abs_diff_eq(Vec3::ZERO, EPSILON));
        assert!(view.transform_point3(target).abs_diff_eq(Vec3::NEG_Z * offset.length(), EPSILON));
    }

    #[test]
    fn finite_projection_maps_near_and_far_to_zero_and_one() {
        let camera = Camera {
            near: 0.5,
            far: Some(64.0),
            ..Camera::default()
        };
        let projection = camera.get_projection(1.5, false);

        assert_depth(projection, 0.5, 0.0);
        assert_depth(projection, 64.0, 1.0);
    }

    #[test]
    fn infinite_projections_map_depth_in_opposite_directions() {
        let camera = Camera {
            near: 0.5,
            ..Camera::default()
        };
        let forward = camera.get_projection(1.5, false);
        let reverse = camera.get_projection(1.5, true);

        assert_depth(forward, 0.5, 0.0);
        assert_depth(forward, 8.0, 0.9375);
        assert_depth(reverse, 0.5, 1.0);
        assert_depth(reverse, 8.0, 0.0625);
    }

    #[test]
    fn camera_projections_map_view_space_up_to_negative_ndc_y() {
        let mut camera = Camera::default();
        let above_center = Vec3::new(0.0, 1.0, -4.0);
        let infinite = camera.get_projection(1.5, false);
        let reverse = camera.get_projection(1.5, true);
        camera.far = Some(64.0);
        let finite = camera.get_projection(1.5, false);

        for projection in [infinite, reverse, finite] {
            assert!(projection.project_point3(above_center).y < 0.0);
        }
    }

    #[test]
    fn walking_stays_level_while_flying_follows_the_camera_pitch() {
        let start = Vec3::new(0.0, 3.0, 0.0);
        let mut walking = Camera {
            position: start,
            ..Camera::default()
        };
        walking.set_rotation(30.0_f32.to_radians(), 0.0, 0.0);
        let mut flying = walking;
        let mut controller = CameraController::new(&walking);
        controller.forward_speed = -4.0;

        controller.update_camera(&mut walking, 0.5);
        assert!(walking.position.abs_diff_eq(start + Vec3::new(0.0, 0.0, -2.0), EPSILON));

        controller.flying = true;
        controller.update_camera(&mut flying, 0.5);
        assert!(flying.position.abs_diff_eq(start + Vec3::new(0.0, 1.0, -3.0_f32.sqrt()), EPSILON));
    }

    #[cfg(feature = "winit")]
    #[test]
    fn mouse_look_responds_while_held_and_clamps_pitch() {
        use winit::event::{DeviceId, ElementState, MouseButton};

        let mut controller = CameraController::new(&Camera::default());
        controller.sensitivity = 0.5;
        let motion = DeviceEvent::MouseMotion { delta: (-30.0, -20.0) };
        let left_button = |state| WindowEvent::MouseInput {
            device_id: DeviceId::dummy(),
            state,
            button: MouseButton::Left,
        };

        controller.update_from_device_event(&motion);
        assert_eq!((controller.yaw, controller.pitch), (0.0, 0.0));

        controller.update_from_window_event(&left_button(ElementState::Pressed));
        controller.update_from_device_event(&motion);
        assert_eq!((controller.yaw, controller.pitch), (15.0, 10.0));
        controller.update_from_device_event(&DeviceEvent::MouseMotion { delta: (0.0, -400.0) });
        assert_eq!(controller.pitch, 89.0);
        controller.update_from_device_event(&DeviceEvent::MouseMotion { delta: (0.0, 400.0) });
        assert_eq!(controller.pitch, -89.0);

        controller.update_from_window_event(&left_button(ElementState::Released));
        controller.update_from_device_event(&motion);
        assert_eq!((controller.yaw, controller.pitch), (15.0, -89.0));
    }
}
