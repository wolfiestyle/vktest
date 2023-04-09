use crate::import::GltfData;
use crate::mesh::MeshId;
use glam::{Affine3A, Mat4, Quat};
use gltf::scene::Transform;
use std::collections::VecDeque;

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
    pub root_nodes: Vec<NodeId>,
    pub name: Option<String>,
}

impl Scene {
    pub fn nodes<'a, V>(&self, gltf: &'a GltfData<V>) -> NodeTreeIter<'a> {
        NodeTreeIter {
            nodes: &gltf.nodes,
            queue: VecDeque::from(self.root_nodes.clone()),
        }
    }

    pub fn nodes_with_parent<'a, V>(&self, gltf: &'a GltfData<V>) -> NodeParentTreeIter<'a> {
        NodeParentTreeIter {
            nodes: &gltf.nodes,
            queue: self.root_nodes.iter().map(|&node| (node, None)).collect(),
        }
    }
}

impl From<gltf::Scene<'_>> for Scene {
    fn from(scene: gltf::Scene) -> Self {
        Self {
            root_nodes: scene.nodes().map(|n| NodeId(n.index())).collect(),
            name: scene.name().map(str::to_string),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodeTreeIter<'a> {
    nodes: &'a [Node],
    queue: VecDeque<NodeId>,
}

impl<'a> Iterator for NodeTreeIter<'a> {
    type Item = &'a Node;

    fn next(&mut self) -> Option<Self::Item> {
        self.queue.pop_front().map(|curr_id| {
            let curr_node = &self.nodes[curr_id.0];
            self.queue.extend(curr_node.children.iter());
            curr_node
        })
    }
}

#[derive(Debug, Clone)]
pub struct NodeParentTreeIter<'a> {
    nodes: &'a [Node],
    queue: VecDeque<(NodeId, Option<NodeId>)>,
}

impl<'a> Iterator for NodeParentTreeIter<'a> {
    type Item = (&'a Node, Option<&'a Node>);

    fn next(&mut self) -> Option<Self::Item> {
        self.queue.pop_front().map(|(curr_id, parent_id)| {
            let curr_node = &self.nodes[curr_id.0];
            let parent_node = parent_id.map(|id| &self.nodes[id.0]);
            self.queue.extend(curr_node.children.iter().map(|&id| (id, Some(curr_id))));
            (curr_node, parent_node)
        })
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

impl Projection {
    pub fn to_matrix(self, viewport_aspect: f32) -> Mat4 {
        match self {
            Self::Ortographic { xmag, ymag, znear, zfar } => {
                // copied from the spec, not sure about this
                let dz = znear - zfar;
                Mat4::from_cols(
                    [1.0 / xmag, 0.0, 0.0, 0.0].into(),
                    [0.0, 1.0 / ymag, 0.0, 0.0].into(),
                    [0.0, 0.0, 2.0 / dz, 0.0].into(),
                    [0.0, 0.0, (zfar + znear) / dz, 1.0].into(),
                )
            }
            Self::Perspective {
                aspect,
                yfov,
                znear,
                zfar: Some(zfar),
            } => {
                let aspect = aspect.unwrap_or(viewport_aspect);
                Mat4::perspective_rh(yfov, aspect, znear, zfar)
            }
            Self::Perspective {
                aspect,
                yfov,
                znear,
                zfar: None,
            } => {
                let aspect = aspect.unwrap_or(viewport_aspect);
                Mat4::perspective_infinite_rh(yfov, aspect, znear)
            }
        }
    }
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
