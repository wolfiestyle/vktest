use crate::import::BufferData;
use crate::material::MaterialId;
use bevy_mikktspace::{generate_tangents, Geometry};
use glam::Vec3A;
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

    fn get_position(&self, index: usize) -> [f32; 3];

    fn get_normal(&self, index: usize) -> [f32; 3];

    fn get_texcoord(&self, index: usize, set: u32) -> [f32; 2];

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

    fn get_position(&self, index: usize) -> [f32; 3] {
        self[index].position
    }

    fn get_normal(&self, index: usize) -> [f32; 3] {
        self[index].normal
    }

    fn get_texcoord(&self, index: usize, set: u32) -> [f32; 2] {
        if set == 0 {
            self[index].texcoord
        } else {
            [0.0; 2]
        }
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
    pub index_offset: u32,
    pub index_count: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub mode: Mode,
    pub material: Option<MaterialId>,
}

impl Submesh {
    pub fn index_range(&self) -> Range<usize> {
        let first = self.index_offset as usize;
        let last = first + self.index_count as usize;
        first..last
    }

    pub fn vertex_range(&self) -> Range<usize> {
        let first = self.vertex_offset as usize;
        let last = first + self.vertex_count as usize;
        first..last
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshData<V> {
    pub vertices: V,
    pub indices: Vec<u32>,
    pub submeshes: Vec<Submesh>,
    pub name: Option<String>,
    pub(crate) vert_offset: usize,
    pub(crate) idx_offset: usize,
}

impl<V: VertexStorage> MeshData<V> {
    pub(crate) fn import_meshes(document: &gltf::Document, buffers: &[BufferData]) -> Vec<Self> {
        document
            .meshes()
            .map(|mesh| {
                let mut data = MeshData {
                    name: mesh.name().map(str::to_string),
                    ..Default::default()
                };
                for prim in mesh.primitives() {
                    data.read_primitive(buffers, &prim);
                }
                data
            })
            .collect()
    }

    fn begin_primitives(&mut self, vert_count: usize, index_count: Option<usize>, mode: Mode, material: Option<MaterialId>) {
        let vert_offset = self.vertices.count();
        self.vertices.set_count(vert_offset + vert_count);
        self.vert_offset = vert_offset;
        let idx_offset = self.indices.len();
        if let Some(idx_count) = index_count {
            self.indices.resize_with(idx_offset + idx_count, Default::default);
            self.submeshes.push(Submesh {
                index_offset: idx_offset as u32,
                index_count: idx_count as u32,
                vertex_offset: vert_offset as u32,
                vertex_count: vert_count as u32,
                mode,
                material,
            });
        } else {
            self.indices.extend(0..vert_count as u32);
            self.submeshes.push(Submesh {
                index_offset: idx_offset as u32,
                index_count: vert_count as u32,
                vertex_offset: vert_offset as u32,
                vertex_count: vert_count as u32,
                mode,
                material,
            });
        }
        self.idx_offset = idx_offset;
    }

    fn end_primitives(&mut self, attribs: VertexAttribs) {
        if let Some(subm) = self.submeshes.last() {
            if subm.mode != Mode::Triangles {
                eprintln!("normal/tangent generation is only supported for triangles (got {:?})", subm.mode);
                return;
            }
            let mut wrapper = GeomWrapper {
                vertices: &mut self.vertices,
                indices: &self.indices[subm.index_range()],
                vert_offset: subm.vertex_offset as usize,
                vert_count: subm.vertex_count as usize,
            };
            if !attribs.normal {
                eprintln!("Generating normals for {} faces", wrapper.num_faces());
                wrapper.generate_normals();
            }
            if !attribs.tangent {
                if attribs.texcoord == 0 {
                    eprintln!("Mesh has missing texcoords, can't generate tangents");
                    return;
                }
                eprintln!("Generating tangents for {} faces", wrapper.num_faces());
                generate_tangents(&mut wrapper);
            }
        }
    }

    fn read_primitive(&mut self, buffers: &[BufferData], prim: &gltf::mesh::Primitive) {
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
        self.begin_primitives(vert_count, idx_count, prim.mode(), prim.material().index().map(MaterialId));

        let reader = prim.reader(|buffer| buffers.get(buffer.index()).map(|buf| buf.data.as_slice()));
        if let Some(iter) = reader.read_positions() {
            for (pos, i) in iter.zip(self.vert_offset..) {
                self.vertices.write_position(i, pos);
            }
        }
        if let Some(iter) = reader.read_normals() {
            for (normal, i) in iter.zip(self.vert_offset..) {
                self.vertices.write_normal(i, normal);
            }
        }
        if let Some(iter) = reader.read_tangents() {
            for (tangent, i) in iter.zip(self.vert_offset..) {
                self.vertices.write_tangent(i, tangent);
            }
        }
        for set in 0..attribs.texcoord {
            if let Some(ty) = reader.read_tex_coords(set) {
                match ty {
                    ReadTexCoords::U8(iter) => {
                        for (texc, i) in iter.zip(self.vert_offset..) {
                            self.vertices.write_texcoord_u8(i, set, texc);
                        }
                    }
                    ReadTexCoords::U16(iter) => {
                        for (texc, i) in iter.zip(self.vert_offset..) {
                            self.vertices.write_texcoord_u16(i, set, texc);
                        }
                    }
                    ReadTexCoords::F32(iter) => {
                        for (texc, i) in iter.zip(self.vert_offset..) {
                            self.vertices.write_texcoord_f32(i, set, texc);
                        }
                    }
                }
            }
        }
        for set in 0..attribs.color {
            if let Some(ty) = reader.read_colors(set) {
                match ty {
                    ReadColors::RgbU8(iter) => {
                        for ([r, g, b], i) in iter.zip(self.vert_offset..) {
                            self.vertices.write_color_u8(i, set, [r, g, b, u8::MAX]);
                        }
                    }
                    ReadColors::RgbU16(iter) => {
                        for ([r, g, b], i) in iter.zip(self.vert_offset..) {
                            self.vertices.write_color_u16(i, set, [r, g, b, u16::MAX]);
                        }
                    }
                    ReadColors::RgbF32(iter) => {
                        for ([r, g, b], i) in iter.zip(self.vert_offset..) {
                            self.vertices.write_color_f32(i, set, [r, g, b, 1.0]);
                        }
                    }
                    ReadColors::RgbaU8(iter) => {
                        for (color, i) in iter.zip(self.vert_offset..) {
                            self.vertices.write_color_u8(i, set, color);
                        }
                    }
                    ReadColors::RgbaU16(iter) => {
                        for (color, i) in iter.zip(self.vert_offset..) {
                            self.vertices.write_color_u16(i, set, color);
                        }
                    }
                    ReadColors::RgbaF32(iter) => {
                        for (color, i) in iter.zip(self.vert_offset..) {
                            self.vertices.write_color_f32(i, set, color);
                        }
                    }
                }
            }
        }
        if let Some(iter) = reader.read_indices() {
            for (index, i) in iter.into_u32().zip(self.idx_offset..) {
                self.indices[i] = index;
            }
        }
        self.end_primitives(attribs);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub usize);

struct GeomWrapper<'a, V> {
    vertices: &'a mut V,
    indices: &'a [u32],
    vert_offset: usize,
    vert_count: usize,
}

impl<V: VertexStorage> Geometry for GeomWrapper<'_, V> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.vertices.get_position(self.indices[face * 3 + vert] as usize)
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.vertices.get_normal(self.indices[face * 3 + vert] as usize)
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.vertices.get_texcoord(self.indices[face * 3 + vert] as usize, 0)
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.vertices.write_tangent(self.indices[face * 3 + vert] as usize, tangent);
    }
}

impl<V: VertexStorage> GeomWrapper<'_, V> {
    fn generate_normals(&mut self) {
        let mut normals = vec![Vec3A::ZERO; self.vert_count];
        for i in (0..self.indices.len()).step_by(3) {
            let ia = self.indices[i] as usize;
            let ib = self.indices[i + 1] as usize;
            let ic = self.indices[i + 2] as usize;
            let posa = Vec3A::from_array(self.vertices.get_position(ia + self.vert_offset));
            let posb = Vec3A::from_array(self.vertices.get_position(ib + self.vert_offset));
            let posc = Vec3A::from_array(self.vertices.get_position(ic + self.vert_offset));
            let normal = (posb - posa).cross(posc - posa);
            normals[ia] += normal;
            normals[ib] += normal;
            normals[ic] += normal;
        }
        for (i, normal) in normals.iter().enumerate() {
            self.vertices
                .write_normal(i + self.vert_offset, normal.normalize_or_zero().to_array());
        }
    }
}
