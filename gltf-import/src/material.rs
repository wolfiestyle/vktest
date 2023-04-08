use gltf::material::{AlphaMode, NormalTexture, OcclusionTexture};
use gltf::texture::{MagFilter, MinFilter, WrappingMode};

#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub color_tex: Option<TextureInfo>,
    pub metallic_roughness_tex: Option<TextureInfo>,
    pub normal_tex: Option<TextureInfo>,
    pub occlusion_tex: Option<TextureInfo>,
    pub emissive_tex: Option<TextureInfo>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

impl From<gltf::Material<'_>> for Material {
    fn from(material: gltf::Material) -> Self {
        let pbr = material.pbr_metallic_roughness();
        let normal_info = material.normal_texture();
        let occl_info = material.occlusion_texture();
        Self {
            base_color: pbr.base_color_factor(),
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            emissive: material.emissive_factor(),
            normal_scale: normal_info.as_ref().map(NormalTexture::scale).unwrap_or(1.0),
            occlusion_strength: occl_info.as_ref().map(OcclusionTexture::strength).unwrap_or(1.0),
            color_tex: pbr.base_color_texture().map(From::from),
            metallic_roughness_tex: pbr.metallic_roughness_texture().map(From::from),
            normal_tex: normal_info.map(From::from),
            occlusion_tex: occl_info.map(From::from),
            emissive_tex: material.emissive_texture().map(From::from),
            alpha_mode: material.alpha_mode(),
            alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
            double_sided: material.double_sided(),
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            metallic: 1.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            color_tex: None,
            metallic_roughness_tex: None,
            normal_tex: None,
            occlusion_tex: None,
            emissive_tex: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureInfo {
    pub id: usize,
    pub uv_set: u32,
}

impl From<gltf::texture::Info<'_>> for TextureInfo {
    fn from(info: gltf::texture::Info) -> Self {
        Self {
            id: info.texture().index(),
            uv_set: info.tex_coord(),
        }
    }
}

impl From<gltf::material::NormalTexture<'_>> for TextureInfo {
    fn from(info: gltf::material::NormalTexture) -> Self {
        Self {
            id: info.texture().index(),
            uv_set: info.tex_coord(),
        }
    }
}

impl From<gltf::material::OcclusionTexture<'_>> for TextureInfo {
    fn from(info: gltf::material::OcclusionTexture) -> Self {
        Self {
            id: info.texture().index(),
            uv_set: info.tex_coord(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageId(pub usize);

#[derive(Debug, Clone, PartialEq)]
pub struct Texture {
    pub image: ImageId,
    pub mag_filter: MagFilter,
    pub min_filter: MinFilter,
    pub wrap_s: WrappingMode,
    pub wrap_t: WrappingMode,
}

impl From<gltf::Texture<'_>> for Texture {
    fn from(tex: gltf::Texture) -> Self {
        let sampler = tex.sampler();
        Self {
            image: ImageId(tex.source().index()),
            mag_filter: sampler.mag_filter().unwrap_or(MagFilter::Linear),
            min_filter: sampler.min_filter().unwrap_or(MinFilter::Linear),
            wrap_s: sampler.wrap_s(),
            wrap_t: sampler.wrap_t(),
        }
    }
}
