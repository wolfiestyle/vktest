use crate::mesh::MeshId;
use glam::{Affine3A, Mat4, Quat};
use gltf::scene::Transform;

#[derive(Debug, Clone)]
pub struct Node {
    pub transform: Affine3A,
    pub mesh: Option<MeshId>,
    pub camera: Option<CameraId>,
    pub children: Vec<NodeId>,
    pub name: Option<String>,
}

impl From<gltf::Node<'_>> for Node {
    fn from(node: gltf::Node) -> Self {
        let transform = match node.transform() {
            Transform::Matrix { matrix } => Affine3A::from_mat4(Mat4::from_cols_array_2d(&matrix)),
            Transform::Decomposed {
                translation,
                rotation,
                scale,
            } => Affine3A::from_scale_rotation_translation(scale.into(), Quat::from_array(rotation), translation.into()),
        };
        Self {
            transform,
            mesh: node.mesh().map(|m| MeshId(m.index())),
            camera: node.camera().map(|c| CameraId(c.index())),
            children: node.children().map(|n| NodeId(n.index())).collect(),
            name: node.name().map(str::to_string),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    pub nodes: Vec<NodeId>,
    pub name: Option<String>,
}

impl From<gltf::Scene<'_>> for Scene {
    fn from(scene: gltf::Scene) -> Self {
        Self {
            nodes: scene.nodes().map(|n| NodeId(n.index())).collect(),
            name: scene.name().map(str::to_string),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    pub projection: Projection,
    pub name: Option<String>,
}

impl From<gltf::Camera<'_>> for Camera {
    fn from(camera: gltf::Camera) -> Self {
        Self {
            projection: camera.projection().into(),
            name: camera.name().map(str::to_string),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CameraId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    Ortographic {
        xmag: f32,
        ymag: f32,
        znear: f32,
        zfar: f32,
    },
    Perspective {
        aspect: Option<f32>,
        yfov: f32,
        znear: f32,
        zfar: Option<f32>,
    },
}

impl From<gltf::camera::Projection<'_>> for Projection {
    fn from(proj: gltf::camera::Projection) -> Self {
        match proj {
            gltf::camera::Projection::Orthographic(ortho) => Projection::Ortographic {
                xmag: ortho.xmag(),
                ymag: ortho.ymag(),
                znear: ortho.znear(),
                zfar: ortho.zfar(),
            },
            gltf::camera::Projection::Perspective(persp) => Projection::Perspective {
                aspect: persp.aspect_ratio(),
                yfov: persp.yfov(),
                znear: persp.znear(),
                zfar: persp.zfar(),
            },
        }
    }
}
