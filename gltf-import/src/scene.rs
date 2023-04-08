use crate::vertex::MeshId;
use gltf::scene::Transform;

#[derive(Debug, Clone)]
pub struct Node {
    pub transform: Transform,
    pub mesh: Option<MeshId>,
    pub camera: Option<CameraId>,
    pub children: Vec<NodeId>,
}

impl From<gltf::Node<'_>> for Node {
    fn from(node: gltf::Node) -> Self {
        Self {
            transform: node.transform(),
            mesh: node.mesh().map(|m| MeshId(m.index())),
            camera: node.camera().map(|c| CameraId(c.index())),
            children: node.children().map(|n| NodeId(n.index())).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    pub nodes: Vec<NodeId>,
}

impl From<gltf::Scene<'_>> for Scene {
    fn from(scene: gltf::Scene) -> Self {
        Self {
            nodes: scene.nodes().map(|n| NodeId(n.index())).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    pub projection: Projection,
}

impl From<gltf::Camera<'_>> for Camera {
    fn from(camera: gltf::Camera) -> Self {
        Self {
            projection: camera.projection().into(),
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
