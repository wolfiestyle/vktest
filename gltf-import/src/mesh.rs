use crate::import::BufferData;
use crate::material::MaterialId;
use bevy_mikktspace::{generate_tangents, Geometry};
use glam::{Vec3, Vec3A};
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

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [i16; 4],
    pub tangent: [i16; 4],
    pub texcoord0: [f32; 2],
    pub texcoord1: [f32; 2],
    pub color: [u8; 4],
}

impl Vertex {
    pub const NUM_UVS: u32 = 2;
    pub const NUM_COLORS: u32 = 1;
}

impl Default for Vertex {
    fn default() -> Self {
        Vertex {
            position: [0.0; 3],
            normal: [0; 4],
            tangent: [0; 4],
            texcoord0: [0.0; 2],
            texcoord1: [0.0; 2],
            color: [u8::MAX; 4],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Primitives {
    pub index_offset: u32,
    pub index_count: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub mode: Mode,
    pub material: Option<MaterialId>,
}

impl Primitives {
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
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub primitives: Vec<Primitives>,
    pub name: Option<String>,
    pub(crate) vert_offset: usize,
    pub(crate) idx_offset: usize,
}

impl MeshData {
    pub(crate) fn import_meshes(document: &gltf::Document, buffers: &[BufferData]) -> Vec<Self> {
        std::thread::scope(|scope| {
            let threads: Vec<_> = document
                .meshes()
                .map(|mesh| {
                    mesh.primitives()
                        .map(|prim| scope.spawn(|| PrimData::read_primitive(buffers, prim)))
                        .collect::<Vec<_>>()
                })
                .collect();

            threads
                .into_iter()
                .zip(document.meshes())
                .map(|(prim_jhs, mesh)| {
                    let prims = prim_jhs.into_iter().map(|jh| jh.join().unwrap());
                    let mut data = MeshData {
                        name: mesh.name().map(str::to_string),
                        ..Default::default()
                    };
                    for primdata in prims {
                        data.append_primitives(primdata);
                    }
                    data
                })
                .collect()
        })
    }

    fn append_primitives(&mut self, prims: PrimData) {
        let vertex_count = prims.positions.len() as u32;
        let index_count = prims.indices.len() as u32;
        let vert_offset = self.vertices.len();
        let idx_offset = self.indices.len();

        let verts = prims
            .positions
            .into_iter()
            .zip(prims.normals)
            .zip(prims.tangents)
            .zip(prims.texcoords0)
            .zip(prims.texcoords1)
            .zip(prims.colors)
            .map(|(((((position, normal), tangent), texcoord0), texcoord1), color)| Vertex {
                position,
                normal: arr_extend(normal.map(f32_to_i16norm), 0),
                tangent: tangent.map(f32_to_i16norm),
                texcoord0,
                texcoord1,
                color,
            });
        self.vertices.extend(verts);
        self.indices.extend(prims.indices);
        self.vert_offset = vert_offset;
        self.idx_offset = idx_offset;

        self.primitives.push(Primitives {
            index_offset: idx_offset as u32,
            index_count,
            vertex_offset: vert_offset as u32,
            vertex_count,
            mode: prims.mode,
            material: prims.material,
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub usize);

struct PrimData {
    mode: Mode,
    material: Option<MaterialId>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tangents: Vec<[f32; 4]>,
    texcoords0: Vec<[f32; 2]>,
    texcoords1: Vec<[f32; 2]>,
    colors: Vec<[u8; 4]>,
    indices: Vec<u32>,
}

impl PrimData {
    fn read_primitive(buffers: &[BufferData], prim: gltf::mesh::Primitive) -> Self {
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
        let mode = prim.mode();
        let material = prim.material().index().map(MaterialId);

        let reader = prim.reader(|buffer| buffers.get(buffer.index()).map(|buf| buf.data.as_slice()));

        let indices: Vec<_> = if let Some(iter) = reader.read_indices() {
            iter.into_u32().collect()
        } else {
            (0..vert_count as u32).collect()
        };
        let positions = if let Some(iter) = reader.read_positions() {
            iter.collect()
        } else {
            vec![[0.0; 3]; vert_count]
        };
        let normals = if let Some(iter) = reader.read_normals() {
            iter.collect()
        } else if mode == Mode::Triangles {
            eprintln!("Generating normals for {vert_count} vertices");
            generate_normals(&positions, &indices)
        } else {
            eprintln!("Can't generate normals for {mode:?}");
            vec![[0.0; 3]; vert_count]
        };
        let texcoords0 = if let Some(iter) = reader.read_tex_coords(0) {
            iter.into_f32().collect()
        } else {
            vec![[0.0; 2]; vert_count]
        };
        let texcoords1 = if let Some(iter) = reader.read_tex_coords(1) {
            iter.into_f32().collect()
        } else {
            vec![[0.0; 2]; vert_count]
        };
        let tangents = if let Some(iter) = reader.read_tangents() {
            iter.collect()
        } else {
            vec![[0.0; 4]; vert_count]
        };
        let colors = if let Some(iter) = reader.read_colors(0) {
            iter.into_rgba_f32().map(linear_to_srgb_approx).collect()
        } else {
            vec![[u8::MAX; 4]; vert_count]
        };

        let mut primdata = Self {
            mode,
            material,
            positions,
            normals,
            tangents,
            texcoords0,
            texcoords1,
            colors,
            indices,
        };

        if mode == Mode::Triangles && !attribs.tangent {
            if attribs.texcoord > 0 {
                eprintln!("Generating MikkTSpace tangents for {vert_count} vertices");
                generate_tangents(&mut primdata);
            } else {
                eprintln!("Generating improvised tangents for {vert_count} vertices");
                primdata.generate_tangents_fallback();
            }
        }
        primdata
    }

    fn generate_tangents_fallback(&mut self) {
        for (i, normal) in self.normals.iter().copied().map(Vec3::from_array).enumerate() {
            let up = if normal.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
            self.tangents[i] = up.cross(normal).normalize_or_zero().extend(1.0).to_array();
        }
    }
}

impl Geometry for PrimData {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.indices[face * 3 + vert] as usize]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.indices[face * 3 + vert] as usize]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.texcoords0[self.indices[face * 3 + vert] as usize]
    }

    fn set_tangent_encoded(&mut self, [x, y, z, w]: [f32; 4], face: usize, vert: usize) {
        self.tangents[self.indices[face * 3 + vert] as usize] = [x, y, z, -w]; // flip bitangent to match glTF coordinates
    }
}

fn generate_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![Vec3A::ZERO; positions.len()];
    for i in (0..indices.len()).step_by(3) {
        let ia = indices[i] as usize;
        let ib = indices[i + 1] as usize;
        let ic = indices[i + 2] as usize;
        let posa = Vec3A::from_array(positions[ia]);
        let posb = Vec3A::from_array(positions[ib]);
        let posc = Vec3A::from_array(positions[ic]);
        let normal = (posb - posa).cross(posc - posa);
        normals[ia] += normal;
        normals[ib] += normal;
        normals[ic] += normal;
    }
    normals.into_iter().map(|vec| vec.normalize_or_zero().to_array()).collect()
}

const I16_MAX_F: f32 = i16::MAX as f32;
const U8_MAX_F: f32 = u8::MAX as f32;

fn f32_to_i16norm(n: f32) -> i16 {
    (n * I16_MAX_F) as i16
}

fn arr_extend<T>([x, y, z]: [T; 3], w: T) -> [T; 4] {
    [x, y, z, w]
}

fn linear_to_srgb_approx([x, y, z, a]: [f32; 4]) -> [u8; 4] {
    let color = Vec3 { x, y, z };
    let srgb = color.powf(1.0 / 2.2).extend(a) * U8_MAX_F;
    srgb.to_array().map(|n| n as u8)
}
