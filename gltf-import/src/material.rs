use gltf::material::AlphaMode;
use gltf::texture::{MagFilter, MinFilter, WrappingMode};

#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub color_tex: Option<TextureInfo>,
    pub metallic_roughness_tex: Option<TextureInfo>,
    pub normal_tex: Option<TextureInfo>,
    pub occlusion_tex: Option<TextureInfo>,
    pub emissive_tex: Option<TextureInfo>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

impl Material {
    pub(crate) fn read(material: gltf::Material) -> Self {
        Self {
            base_color: material.pbr_metallic_roughness().base_color_factor(),
            metallic: material.pbr_metallic_roughness().metallic_factor(),
            roughness: material.pbr_metallic_roughness().roughness_factor(),
            emissive: material.emissive_factor(),
            color_tex: material.pbr_metallic_roughness().base_color_texture().map(TextureInfo::read),
            metallic_roughness_tex: material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
                .map(TextureInfo::read),
            normal_tex: material.normal_texture().map(TextureInfo::read_normal),
            occlusion_tex: material.occlusion_texture().map(TextureInfo::read_occlusion),
            emissive_tex: material.emissive_texture().map(TextureInfo::read),
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
    pub index: usize,
    pub uv_set: u32,
}

impl TextureInfo {
    fn read(info: gltf::texture::Info) -> Self {
        Self {
            index: info.texture().index(),
            uv_set: info.tex_coord(),
        }
    }

    fn read_normal(info: gltf::material::NormalTexture) -> Self {
        Self {
            index: info.texture().index(),
            uv_set: info.tex_coord(),
        }
    }

    fn read_occlusion(info: gltf::material::OcclusionTexture) -> Self {
        Self {
            index: info.texture().index(),
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

impl Texture {
    pub(crate) fn read(tex: gltf::Texture) -> Self {
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
