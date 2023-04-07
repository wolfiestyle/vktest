#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct VertexAttribs {
    pub position: bool,
    pub normal: bool,
    pub tangent: bool,
    pub texcoord: u32,
    pub color: u32,
}

pub trait VertexOutput {
    fn init(&mut self, vert_count: usize, index_count: Option<usize>, attrib_present: VertexAttribs);

    fn write_positions(&mut self, data: impl Iterator<Item = [f32; 3]>);

    fn write_normals(&mut self, data: impl Iterator<Item = [f32; 3]>);

    fn write_tangents(&mut self, data: impl Iterator<Item = [f32; 4]>);

    fn write_texcoords_f32(&mut self, set: u32, data: impl Iterator<Item = [f32; 2]>);

    fn write_texcoords_u8(&mut self, set: u32, data: impl Iterator<Item = [u8; 2]>) {
        self.write_texcoords_f32(set, data.map(|t| t.map(|n| n as f32 / 255.0)))
    }

    fn write_texcoords_u16(&mut self, set: u32, data: impl Iterator<Item = [u16; 2]>) {
        self.write_texcoords_f32(set, data.map(|t| t.map(|n| n as f32 / 65535.0)))
    }

    fn write_colors_f32(&mut self, set: u32, data: impl Iterator<Item = [f32; 4]>);

    fn write_colors_u8(&mut self, set: u32, data: impl Iterator<Item = [u8; 4]>) {
        self.write_colors_f32(set, data.map(|c| c.map(|n| n as f32 / 255.0)))
    }

    fn write_colors_u16(&mut self, set: u32, data: impl Iterator<Item = [u16; 4]>) {
        self.write_colors_f32(set, data.map(|c| c.map(|n| n as f32 / 65536.0)))
    }

    fn write_indices(&mut self, data: impl Iterator<Item = u32>);

    fn finish(&mut self);
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

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub attribs: VertexAttribs,
    vert_offset: usize,
    idx_offset: usize,
}

impl MeshData {
    pub fn new() -> Self {
        Default::default()
    }
}

impl VertexOutput for MeshData {
    fn init(&mut self, vert_count: usize, index_count: Option<usize>, attrib_present: VertexAttribs) {
        let vert_offset = self.vertices.len();
        self.vertices.resize_with(vert_offset + vert_count, Default::default);
        self.vert_offset = vert_offset;
        let idx_offset = self.indices.len();
        if let Some(idx_count) = index_count {
            self.indices.resize_with(idx_offset + idx_count, Default::default);
        } else {
            let first = vert_offset as u32;
            let last = first + vert_count as u32;
            self.indices.extend(first..last);
        }
        self.idx_offset = idx_offset;
        self.attribs = attrib_present;
    }

    fn write_positions(&mut self, data: impl Iterator<Item = [f32; 3]>) {
        for (pos, i) in data.zip(self.vert_offset..) {
            self.vertices[i].position = pos;
        }
    }

    fn write_normals(&mut self, data: impl Iterator<Item = [f32; 3]>) {
        for (normal, i) in data.zip(self.vert_offset..) {
            self.vertices[i].normal = normal;
        }
    }

    fn write_tangents(&mut self, data: impl Iterator<Item = [f32; 4]>) {
        for (tangent, i) in data.zip(self.vert_offset..) {
            self.vertices[i].tangent = tangent;
        }
    }

    fn write_texcoords_f32(&mut self, set: u32, data: impl Iterator<Item = [f32; 2]>) {
        if set == 0 {
            for (texc, i) in data.zip(self.vert_offset..) {
                self.vertices[i].texcoord = texc;
            }
        }
    }

    fn write_colors_f32(&mut self, set: u32, data: impl Iterator<Item = [f32; 4]>) {
        if set == 0 {
            for (color, i) in data.zip(self.vert_offset..) {
                self.vertices[i].color = color;
            }
        }
    }

    fn write_indices(&mut self, data: impl Iterator<Item = u32>) {
        let base = self.vert_offset as u32;
        for (index, i) in data.zip(self.idx_offset..) {
            self.indices[i] = index + base;
        }
    }

    fn finish(&mut self) {}
}
