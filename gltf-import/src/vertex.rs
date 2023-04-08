use crate::import::{BufferData, GltfData};
use crate::material::MaterialId;
use gltf::mesh::util::{ReadColors, ReadTexCoords};
use gltf::mesh::Mode;
use gltf::Semantic;
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

#[repr(C)]
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
    pub index_range: Range<usize>,
    pub mode: Mode,
    pub material: Option<MaterialId>,
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
    pub fn read_meshes(gltf: &GltfData) -> Vec<Self> {
        gltf.document
            .meshes()
            .map(|mesh| {
                let mut data = MeshData::default();
                for prim in mesh.primitives() {
                    Self::read_primitive(&gltf.buffers, &prim, &mut data);
                }
                data
            })
            .collect()
    }

    pub(crate) fn begin_primitives(
        &mut self, vert_count: usize, index_count: Option<usize>, attribs: VertexAttribs, mode: Mode, material: Option<MaterialId>,
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
                material,
            });
        } else {
            let first = vert_offset as u32;
            let last = first + vert_count as u32;
            self.indices.extend(first..last);
            self.submeshes.push(Submesh {
                index_range: idx_offset..idx_offset + vert_count,
                mode,
                material,
            });
        }
        self.idx_offset = idx_offset;
        self.attribs = attribs; //FIXME: this could change between submeshes
    }

    fn read_primitive(buffers: &BufferData, prim: &gltf::mesh::Primitive, output: &mut MeshData<V>) {
        let mut attribs = VertexAttribs::default();
        let mut vert_count = 0;
        for (semantic, accessor) in prim.attributes() {
            match semantic {
                Semantic::Positions => attribs.position = true,
                Semantic::Normals => attribs.normal = true,
                Semantic::Tangents => attribs.tangent = true,
                Semantic::TexCoords(n) => attribs.texcoord = attribs.texcoord.max(n + 1),
                Semantic::Colors(n) => attribs.color = attribs.color.max(n + 1),
                _ => (),
            }
            vert_count = vert_count.max(accessor.count());
        }
        let idx_count = prim.indices().map(|acc| acc.count());
        output.begin_primitives(vert_count, idx_count, attribs, prim.mode(), prim.material().index().map(MaterialId));

        let reader = prim.reader(|buffer| buffers.0.get(buffer.index()).map(Vec::as_slice));
        if let Some(iter) = reader.read_positions() {
            for (pos, i) in iter.zip(output.vert_offset..) {
                output.vertices.write_position(i, pos);
            }
        }
        if let Some(iter) = reader.read_normals() {
            for (normal, i) in iter.zip(output.vert_offset..) {
                output.vertices.write_normal(i, normal);
            }
        }
        if let Some(iter) = reader.read_tangents() {
            for (tangent, i) in iter.zip(output.vert_offset..) {
                output.vertices.write_tangent(i, tangent);
            }
        }
        for set in 0..attribs.texcoord {
            if let Some(ty) = reader.read_tex_coords(set) {
                match ty {
                    ReadTexCoords::U8(iter) => {
                        for (texc, i) in iter.zip(output.vert_offset..) {
                            output.vertices.write_texcoord_u8(i, set, texc);
                        }
                    }
                    ReadTexCoords::U16(iter) => {
                        for (texc, i) in iter.zip(output.vert_offset..) {
                            output.vertices.write_texcoord_u16(i, set, texc);
                        }
                    }
                    ReadTexCoords::F32(iter) => {
                        for (texc, i) in iter.zip(output.vert_offset..) {
                            output.vertices.write_texcoord_f32(i, set, texc);
                        }
                    }
                }
            }
        }
        for set in 0..attribs.color {
            if let Some(ty) = reader.read_colors(set) {
                match ty {
                    ReadColors::RgbU8(iter) => {
                        for ([r, g, b], i) in iter.zip(output.vert_offset..) {
                            output.vertices.write_color_u8(i, set, [r, g, b, u8::MAX]);
                        }
                    }
                    ReadColors::RgbU16(iter) => {
                        for ([r, g, b], i) in iter.zip(output.vert_offset..) {
                            output.vertices.write_color_u16(i, set, [r, g, b, u16::MAX]);
                        }
                    }
                    ReadColors::RgbF32(iter) => {
                        for ([r, g, b], i) in iter.zip(output.vert_offset..) {
                            output.vertices.write_color_f32(i, set, [r, g, b, 1.0]);
                        }
                    }
                    ReadColors::RgbaU8(iter) => {
                        for (color, i) in iter.zip(output.vert_offset..) {
                            output.vertices.write_color_u8(i, set, color);
                        }
                    }
                    ReadColors::RgbaU16(iter) => {
                        for (color, i) in iter.zip(output.vert_offset..) {
                            output.vertices.write_color_u16(i, set, color);
                        }
                    }
                    ReadColors::RgbaF32(iter) => {
                        for (color, i) in iter.zip(output.vert_offset..) {
                            output.vertices.write_color_f32(i, set, color);
                        }
                    }
                }
            }
        }
        if let Some(iter) = reader.read_indices() {
            let base = output.vert_offset as u32;
            for (index, i) in iter.into_u32().zip(output.idx_offset..) {
                output.indices[i] = index + base;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub usize);
