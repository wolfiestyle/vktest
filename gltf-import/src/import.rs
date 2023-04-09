use crate::material::{ImageId, Material, MaterialId, Texture, TextureInfo};
use crate::mesh::{MeshData, MeshId, Vertex, VertexStorage};
use crate::scene::{Camera, CameraId, Node, NodeId, Scene};
use crate::types::*;
use crate::uri::Uri;
use gltf::Document;
use image::{DynamicImage, ImageFormat};
use std::borrow::Cow;
use std::fs::File;
use std::io::BufReader;
use std::ops;
use std::path::Path;
use std::sync::Arc;

pub type Gltf = GltfData<Vec<Vertex>>;

#[derive(Debug, Clone)]
pub struct GltfData<V> {
    pub document: Document,
    pub buffers: Vec<BufferData>,
    pub images: Vec<Image>,
    pub meshes: Vec<MeshData<V>>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub nodes: Vec<Node>,
    pub scenes: Vec<Scene>,
    pub cameras: Vec<Camera>,
}

impl<V: VertexStorage> GltfData<V> {
    pub fn from_file(path: impl AsRef<Path>) -> ImportResult<Self> {
        let path = path.as_ref();
        let base_path = path.parent().unwrap_or_else(|| Path::new("."));
        let file = File::open(path).map_err(|err| ImportError::Io(err, path.into()))?;
        let gltf = gltf::Gltf::from_reader(BufReader::new(file))?;
        Self::import_gltf(gltf, Some(base_path))
    }

    pub fn from_memory(bytes: &[u8]) -> ImportResult<Self> {
        let gltf = gltf::Gltf::from_slice(bytes)?;
        Self::import_gltf(gltf, None)
    }

    fn import_gltf(gltf: gltf::Gltf, base_path: Option<&Path>) -> ImportResult<Self> {
        let buffers = BufferData::import_buffers(&gltf.document, gltf.blob, base_path)?;
        let images = Image::import_images(&gltf.document, &buffers, base_path);
        let meshes = MeshData::import_meshes(&gltf.document, &buffers);
        let materials = gltf.document.materials().map(From::from).collect();
        let textures = gltf.document.textures().map(From::from).collect();
        let nodes = gltf.document.nodes().map(From::from).collect();
        let scenes = gltf.document.scenes().map(From::from).collect();
        let cameras = gltf.document.cameras().map(From::from).collect();

        Ok(Self {
            document: gltf.document,
            buffers,
            images,
            meshes,
            materials,
            textures,
            nodes,
            scenes,
            cameras,
        })
    }
}

impl<V> ops::Index<ImageId> for GltfData<V> {
    type Output = Image;

    #[inline]
    fn index(&self, id: ImageId) -> &Self::Output {
        &self.images[id.0]
    }
}

impl<V> ops::Index<MeshId> for GltfData<V> {
    type Output = MeshData<V>;

    #[inline]
    fn index(&self, id: MeshId) -> &Self::Output {
        &self.meshes[id.0]
    }
}

impl<V> ops::Index<MaterialId> for GltfData<V> {
    type Output = Material;

    #[inline]
    fn index(&self, id: MaterialId) -> &Self::Output {
        &self.materials[id.0]
    }
}

impl<V> ops::Index<TextureInfo> for GltfData<V> {
    type Output = Texture;

    #[inline]
    fn index(&self, info: TextureInfo) -> &Self::Output {
        &self.textures[info.id]
    }
}

impl<V> ops::Index<NodeId> for GltfData<V> {
    type Output = Node;

    #[inline]
    fn index(&self, id: NodeId) -> &Self::Output {
        &self.nodes[id.0]
    }
}

impl<V> ops::Index<CameraId> for GltfData<V> {
    type Output = Camera;

    #[inline]
    fn index(&self, id: CameraId) -> &Self::Output {
        &self.cameras[id.0]
    }
}

#[derive(Debug, Clone)]
pub struct BufferData {
    pub data: Vec<u8>,
    pub name: Option<String>,
}

impl BufferData {
    fn import_buffers(document: &Document, mut blob: Option<Vec<u8>>, base_path: Option<&Path>) -> ImportResult<Vec<Self>> {
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
                Ok(Self {
                    data,
                    name: buffer.name().map(str::to_string),
                })
            })
            .collect::<Result<_, _>>()
    }

    pub fn view_slice<'a>(buffers: &'a [Self], view: &gltf::buffer::View) -> &'a [u8] {
        let buffer = &buffers[view.buffer().index()];
        let offset = view.offset();
        let size = view.length();
        &buffer.data[offset..offset + size]
    }
}

#[derive(Debug, Clone)]
pub struct Image {
    pub data: ImageData,
    pub mime_type: Option<String>,
    pub name: Option<String>,
}

impl Image {
    fn import_images(document: &Document, buffers: &[BufferData], base_path: Option<&Path>) -> Vec<Self> {
        std::thread::scope(|scope| {
            let threads: Vec<_> = document
                .images()
                .map(|image_src| {
                    scope.spawn(move || {
                        let (data, mime_type) = ImageData::load(&image_src, buffers, base_path);
                        Image {
                            data,
                            name: image_src.name().map(str::to_string),
                            mime_type,
                        }
                    })
                })
                .collect();
            threads.into_iter().map(|jh| jh.join().unwrap()).collect()
        })
    }
}

#[derive(Debug, Clone)]
pub enum ImageData {
    Decoded(DynamicImage),
    Raw(Vec<u8>),
    Uri(String),
    Missing(Arc<ImportError>),
}

impl ImageData {
    fn load(image_src: &gltf::image::Image, buffers: &[BufferData], base_path: Option<&Path>) -> (Self, Option<String>) {
        use gltf::image::Source;

        match image_src.source() {
            Source::Uri { uri, .. } => {
                let uri = match Uri::parse(uri) {
                    Ok(uri) => uri,
                    Err(err) => return (Self::Missing(err.into()), None),
                };
                let mtype = uri.media_type();
                let image = match uri.read_contents(base_path) {
                    Ok(data) => Self::decode(mtype, data.into()),
                    Err(ImportError::UnsupportedUri(uri)) => Self::Uri(uri),
                    Err(err) => Self::Missing(err.into()),
                };
                (image, mtype.map(str::to_string))
            }
            Source::View { view, mime_type } => {
                let data = BufferData::view_slice(buffers, &view);
                let image = Self::decode(Some(mime_type), data.into());
                (image, Some(mime_type.to_string()))
            }
        }
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
            Self::Raw(data.into_owned())
        }
    }
}
