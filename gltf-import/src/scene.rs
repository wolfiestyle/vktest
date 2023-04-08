use crate::vertex::MeshId;
use gltf::scene::Transform;

#[derive(Debug, Clone)]
pub struct Node {
    pub transform: Transform,
    pub mesh: Option<MeshId>,
    pub camera: Option<CameraId>,
    pub children: Vec<NodeId>,
}

impl Node {
    pub(crate) fn read(node: gltf::Node) -> Self {
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

impl Scene {
    pub(crate) fn read(scene: gltf::Scene) -> Self {
        Self {
            nodes: scene.nodes().map(|n| NodeId(n.index())).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    pub projection: Projection,
}

impl Camera {
    pub(crate) fn read(camera: gltf::Camera) -> Self {
        Self {
            projection: Projection::read(camera.projection()),
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

impl Projection {
    pub(crate) fn read(proj: gltf::camera::Projection) -> Self {
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
