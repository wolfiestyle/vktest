use crate::types::*;
use crate::uri::Uri;
use crate::vertex::{VertexAttribs, VertexOutput};
use gltf::mesh::util::{ReadColors, ReadTexCoords};
use gltf::{Document, Gltf, Semantic};
use image::{DynamicImage, ImageFormat};
use std::borrow::Cow;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct GltfData {
    pub document: Document,
    pub buffers: BufferData,
    pub images: Vec<ImageData>,
}

impl GltfData {
    pub fn from_file(path: impl AsRef<Path>) -> ImportResult<Self> {
        let path = path.as_ref();
        let base_path = path.parent().unwrap_or_else(|| Path::new("."));
        let file = File::open(path).map_err(|err| ImportError::Io(err, path.into()))?;
        let gltf = Gltf::from_reader(BufReader::new(file))?;
        Self::import_gltf(gltf, Some(base_path))
    }

    pub fn from_memory(bytes: &[u8]) -> ImportResult<Self> {
        let gltf = Gltf::from_slice(bytes)?;
        Self::import_gltf(gltf, None)
    }

    fn import_gltf(gltf: Gltf, base_path: Option<&Path>) -> ImportResult<Self> {
        let buffers = BufferData::import_buffers(&gltf.document, gltf.blob, base_path)?;
        let images = ImageData::import_images(&gltf.document, &buffers, base_path);
        Ok(Self {
            document: gltf.document,
            buffers,
            images,
        })
    }

    pub fn read_vertices<T: VertexOutput>(&self, prim: &gltf::mesh::Primitive, output: &mut T) {
        let mut attribs = VertexAttribs::default();
        let mut count = 0;
        for (semantic, accessor) in prim.attributes() {
            match semantic {
                Semantic::Positions => attribs.position = true,
                Semantic::Normals => attribs.normal = true,
                Semantic::Tangents => attribs.tangent = true,
                Semantic::TexCoords(n) => attribs.texcoord = attribs.texcoord.max(n + 1),
                Semantic::Colors(n) => attribs.color = attribs.color.max(n + 1),
                _ => (),
            }
            count = count.max(accessor.count());
        }
        output.init(count, attribs);

        let reader = prim.reader(|buffer| self.buffers.0.get(buffer.index()).map(Vec::as_slice));
        if let Some(iter) = reader.read_positions() {
            output.write_positions(iter);
        }
        if let Some(iter) = reader.read_normals() {
            output.write_normals(iter);
        }
        if let Some(iter) = reader.read_tangents() {
            output.write_tangents(iter);
        }
        for set in 0..attribs.texcoord {
            if let Some(ty) = reader.read_tex_coords(set) {
                match ty {
                    ReadTexCoords::U8(iter) => output.write_texcoords_u8(set, iter),
                    ReadTexCoords::U16(iter) => output.write_texcoords_u16(set, iter),
                    ReadTexCoords::F32(iter) => output.write_texcoords_f32(set, iter),
                }
            }
        }
        for set in 0..attribs.color {
            if let Some(ty) = reader.read_colors(set) {
                match ty {
                    ReadColors::RgbU8(iter) => output.write_colors_u8(set, iter.map(|[r, g, b]| [r, g, b, u8::MAX])),
                    ReadColors::RgbU16(iter) => output.write_colors_u16(set, iter.map(|[r, g, b]| [r, g, b, u16::MAX])),
                    ReadColors::RgbF32(iter) => output.write_colors_f32(set, iter.map(|[r, g, b]| [r, g, b, 1.0])),
                    ReadColors::RgbaU8(iter) => output.write_colors_u8(set, iter),
                    ReadColors::RgbaU16(iter) => output.write_colors_u16(set, iter),
                    ReadColors::RgbaF32(iter) => output.write_colors_f32(set, iter),
                }
            }
        }
        output.finish();
    }
}

#[derive(Debug, Clone)]
pub struct BufferData(pub Vec<Vec<u8>>);

impl BufferData {
    fn import_buffers(document: &Document, mut blob: Option<Vec<u8>>, base_path: Option<&Path>) -> ImportResult<Self> {
        use gltf::buffer::Source;

        document
            .buffers()
            .map(|buffer| {
                let data = match buffer.source() {
                    Source::Uri(uri) => Uri::parse(uri)?.read_contents(base_path)?,
                    Source::Bin => blob.take().ok_or(ImportError::MissingBlob)?,
                };
                if data.len() < buffer.length() {
                    return Err(ImportError::BufferSize {
                        id: buffer.index(),
                        actual: data.len(),
                        expected: buffer.length(),
                    });
                }
                Ok(data)
            })
            .collect::<Result<_, _>>()
            .map(Self)
    }

    pub fn view_slice(&self, view: &gltf::buffer::View) -> &[u8] {
        let buffer = &self.0[view.buffer().index()];
        let offset = view.offset();
        let size = view.length();
        &buffer[offset..offset + size]
    }
}

#[derive(Debug, Clone)]
pub enum ImageData {
    Decoded(DynamicImage),
    Raw { data: Vec<u8>, mime_type: Option<String> },
    Uri(String),
    Missing(Arc<ImportError>),
}

impl ImageData {
    fn import_images(document: &Document, buffers: &BufferData, base_path: Option<&Path>) -> Vec<Self> {
        use gltf::image::Source;

        document
            .images()
            .map(|image_src| match image_src.source() {
                Source::Uri { uri, .. } => {
                    let uri = match Uri::parse(uri) {
                        Ok(uri) => uri,
                        Err(err) => return Self::Missing(err.into()),
                    };
                    let mtype = uri.media_type();
                    match uri.read_contents(base_path) {
                        Ok(data) => Self::decode(mtype, data.into()),
                        Err(ImportError::UnsupportedUri(uri)) => Self::Uri(uri),
                        Err(err) => Self::Missing(err.into()),
                    }
                }
                Source::View { view, mime_type } => {
                    let data = buffers.view_slice(&view);
                    Self::decode(Some(mime_type), data.into())
                }
            })
            .collect()
    }

    fn decode(mime_type: Option<&str>, data: Cow<[u8]>) -> Self {
        let format = match mime_type {
            Some("image/png") => Some(ImageFormat::Png),
            Some("image/jpeg") => Some(ImageFormat::Jpeg),
            Some("image/webp") => Some(ImageFormat::WebP),
            Some("image/gif") => Some(ImageFormat::Gif),
            Some("image/bmp") => Some(ImageFormat::Bmp),
            _ => image::guess_format(&data).ok(),
        };
        if let Some(fmt) = format {
            match image::load_from_memory_with_format(&data, fmt) {
                Ok(image) => Self::Decoded(image),
                Err(err) => Self::Missing(ImportError::Image(err).into()),
            }
        } else {
            Self::Raw {
                data: data.into_owned(),
                mime_type: mime_type.map(str::to_string),
            }
        }
    }
}
