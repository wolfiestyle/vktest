use crate::import::BufferData;
use gltf::mesh::Mode;
use gltf::Document;
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct VertexAttribs {
    pub position: bool,
    pub normal: bool,
    pub tangent: bool,
    pub texcoord: u32,
    pub color: u32,
}

pub trait VertexStorage: Default {
    fn count(&self) -> usize;

    fn set_count(&mut self, new_size: usize);

    fn write_position(&mut self, index: usize, value: [f32; 3]);

    fn write_normal(&mut self, index: usize, value: [f32; 3]);

    fn write_tangent(&mut self, index: usize, value: [f32; 4]);

    fn write_texcoord_f32(&mut self, index: usize, set: u32, value: [f32; 2]);

    fn write_texcoord_u8(&mut self, index: usize, set: u32, value: [u8; 2]);

    fn write_texcoord_u16(&mut self, index: usize, set: u32, value: [u16; 2]);

    fn write_color_f32(&mut self, index: usize, set: u32, value: [f32; 4]);

    fn write_color_u8(&mut self, index: usize, set: u32, value: [u8; 4]);

    fn write_color_u16(&mut self, index: usize, set: u32, value: [u16; 4]);
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub texcoord: [f32; 2],
    pub color: [f32; 4],
}

impl Default for Vertex {
    fn default() -> Self {
        Vertex {
            position: [0.0; 3],
            normal: [0.0; 3],
            tangent: [0.0; 4],
            texcoord: [0.0; 2],
            color: [1.0; 4],
        }
    }
}

impl VertexStorage for Vec<Vertex> {
    fn count(&self) -> usize {
        self.len()
    }

    fn set_count(&mut self, new_count: usize) {
        self.resize_with(new_count, Default::default)
    }

    fn write_position(&mut self, index: usize, value: [f32; 3]) {
        self[index].position = value;
    }

    fn write_normal(&mut self, index: usize, value: [f32; 3]) {
        self[index].normal = value;
    }

    fn write_tangent(&mut self, index: usize, value: [f32; 4]) {
        self[index].tangent = value;
    }

    fn write_texcoord_f32(&mut self, index: usize, set: u32, value: [f32; 2]) {
        if set == 0 {
            self[index].texcoord = value;
        }
    }

    fn write_texcoord_u8(&mut self, index: usize, set: u32, value: [u8; 2]) {
        self.write_texcoord_f32(index, set, value.map(|n| n as f32 / 255.0))
    }

    fn write_texcoord_u16(&mut self, index: usize, set: u32, value: [u16; 2]) {
        self.write_texcoord_f32(index, set, value.map(|n| n as f32 / 65535.0))
    }

    fn write_color_f32(&mut self, index: usize, set: u32, value: [f32; 4]) {
        if set == 0 {
            self[index].color = value;
        }
    }

    fn write_color_u8(&mut self, index: usize, set: u32, value: [u8; 4]) {
        self.write_color_f32(index, set, value.map(|n| n as f32 / 255.0))
    }

    fn write_color_u16(&mut self, index: usize, set: u32, value: [u16; 4]) {
        self.write_color_f32(index, set, value.map(|n| n as f32 / 65535.0))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submesh {
    index_range: Range<usize>,
    mode: Mode,
    material_id: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshData<V> {
    pub vertices: V,
    pub indices: Vec<u32>,
    pub attribs: VertexAttribs,
    pub submeshes: Vec<Submesh>,
    pub(crate) vert_offset: usize,
    pub(crate) idx_offset: usize,
}

impl<V: VertexStorage> MeshData<V> {
    pub(crate) fn import_meshes(document: &Document, buffers: &BufferData) -> Vec<Self> {
        document
            .meshes()
            .map(|mesh| {
                let mut data = MeshData::default();
                for prim in mesh.primitives() {
                    buffers.read_primitive(&prim, &mut data);
                }
                data
            })
            .collect()
    }

    pub(crate) fn begin_primitives(
        &mut self, vert_count: usize, index_count: Option<usize>, attribs: VertexAttribs, mode: Mode, material_id: Option<usize>,
    ) {
        let vert_offset = self.vertices.count();
        self.vertices.set_count(vert_offset + vert_count);
        self.vert_offset = vert_offset;
        let idx_offset = self.indices.len();
        if let Some(idx_count) = index_count {
            let idx_end = idx_offset + idx_count;
            self.indices.resize_with(idx_end, Default::default);
            self.submeshes.push(Submesh {
                index_range: idx_offset..idx_end,
                mode,
                material_id,
            });
        } else {
            let first = vert_offset as u32;
            let last = first + vert_count as u32;
            self.indices.extend(first..last);
            self.submeshes.push(Submesh {
                index_range: idx_offset..idx_offset + vert_count,
                mode,
                material_id,
            });
        }
        self.idx_offset = idx_offset;
        self.attribs = attribs; //FIXME: this could change between submeshes
    }
}
