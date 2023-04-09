use ash::vk;
use memoffset::{offset_of, offset_of_tuple};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct UNorm<T>(T);

impl<T> From<T> for UNorm<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self(value)
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SNorm<T>(T);

impl<T> From<T> for SNorm<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self(value)
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SRgb<T>(T);

impl<T> From<T> for SRgb<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self(value)
    }
}

pub trait TypeFormat: Copy {
    const VK_FORMAT: vk::Format;
    const VK_WIDTH: u32;
}

macro_rules! impl_format {
    ($type:ty, $width:expr, $val:expr) => {
        impl TypeFormat for $type {
            const VK_FORMAT: vk::Format = $val;
            const VK_WIDTH: u32 = $width;
        }
    };
}

impl_format!(f32, 1, vk::Format::R32_SFLOAT);
impl_format!([f32; 1], 1, vk::Format::R32_SFLOAT);
impl_format!([f32; 2], 1, vk::Format::R32G32_SFLOAT);
impl_format!([f32; 3], 1, vk::Format::R32G32B32_SFLOAT);
impl_format!([f32; 4], 1, vk::Format::R32G32B32A32_SFLOAT);

impl_format!(u8, 1, vk::Format::R8_UINT);
impl_format!([u8; 1], 1, vk::Format::R8_UINT);
impl_format!([u8; 2], 1, vk::Format::R8G8_UINT);
impl_format!([u8; 3], 1, vk::Format::R8G8B8_UINT);
impl_format!([u8; 4], 1, vk::Format::R8G8B8A8_UINT);

impl_format!(u16, 1, vk::Format::R16_UINT);
impl_format!([u16; 1], 1, vk::Format::R16_UINT);
impl_format!([u16; 2], 1, vk::Format::R16G16_UINT);
impl_format!([u16; 3], 1, vk::Format::R16G16B16_UINT);
impl_format!([u16; 4], 1, vk::Format::R16G16B16A16_UINT);

impl_format!(u32, 1, vk::Format::R32_UINT);
impl_format!([u32; 1], 1, vk::Format::R32_UINT);
impl_format!([u32; 2], 1, vk::Format::R32G32_UINT);
impl_format!([u32; 3], 1, vk::Format::R32G32B32_UINT);
impl_format!([u32; 4], 1, vk::Format::R32G32B32A32_UINT);

impl_format!(i8, 1, vk::Format::R8_SINT);
impl_format!([i8; 1], 1, vk::Format::R8_SINT);
impl_format!([i8; 2], 1, vk::Format::R8G8_SINT);
impl_format!([i8; 3], 1, vk::Format::R8G8B8_SINT);
impl_format!([i8; 4], 1, vk::Format::R8G8B8A8_SINT);

impl_format!(i16, 1, vk::Format::R16_SINT);
impl_format!([i16; 1], 1, vk::Format::R16_SINT);
impl_format!([i16; 2], 1, vk::Format::R16G16_SINT);
impl_format!([i16; 3], 1, vk::Format::R16G16B16_SINT);
impl_format!([i16; 4], 1, vk::Format::R16G16B16A16_SINT);

impl_format!(i32, 1, vk::Format::R32_SINT);
impl_format!([i32; 1], 1, vk::Format::R32_SINT);
impl_format!([i32; 2], 1, vk::Format::R32G32_SINT);
impl_format!([i32; 3], 1, vk::Format::R32G32B32_SINT);
impl_format!([i32; 4], 1, vk::Format::R32G32B32A32_SINT);

impl_format!(UNorm<u8>, 1, vk::Format::R8_UNORM);
impl_format!(UNorm<[u8; 1]>, 1, vk::Format::R8_UNORM);
impl_format!(UNorm<[u8; 2]>, 1, vk::Format::R8G8_UNORM);
impl_format!(UNorm<[u8; 3]>, 1, vk::Format::R8G8B8_UNORM);
impl_format!(UNorm<[u8; 4]>, 1, vk::Format::R8G8B8A8_UNORM);

impl_format!(UNorm<u16>, 1, vk::Format::R16_UNORM);
impl_format!(UNorm<[u16; 1]>, 1, vk::Format::R16_UNORM);
impl_format!(UNorm<[u16; 2]>, 1, vk::Format::R16G16_UNORM);
impl_format!(UNorm<[u16; 3]>, 1, vk::Format::R16G16B16_UNORM);
impl_format!(UNorm<[u16; 4]>, 1, vk::Format::R16G16B16A16_UNORM);

impl_format!(SNorm<i8>, 1, vk::Format::R8_SNORM);
impl_format!(SNorm<[i8; 1]>, 1, vk::Format::R8_SNORM);
impl_format!(SNorm<[i8; 2]>, 1, vk::Format::R8G8_SNORM);
impl_format!(SNorm<[i8; 3]>, 1, vk::Format::R8G8B8_SNORM);
impl_format!(SNorm<[i8; 4]>, 1, vk::Format::R8G8B8A8_SNORM);

impl_format!(SNorm<i16>, 1, vk::Format::R16_SNORM);
impl_format!(SNorm<[i16; 1]>, 1, vk::Format::R16_SNORM);
impl_format!(SNorm<[i16; 2]>, 1, vk::Format::R16G16_SNORM);
impl_format!(SNorm<[i16; 3]>, 1, vk::Format::R16G16B16_SNORM);
impl_format!(SNorm<[i16; 4]>, 1, vk::Format::R16G16B16A16_SNORM);

impl_format!(SRgb<u8>, 1, vk::Format::R8_SRGB);
impl_format!(SRgb<[u8; 1]>, 1, vk::Format::R8_SRGB);
impl_format!(SRgb<[u8; 2]>, 1, vk::Format::R8G8_SRGB);
impl_format!(SRgb<[u8; 3]>, 1, vk::Format::R8G8B8_SRGB);
impl_format!(SRgb<[u8; 4]>, 1, vk::Format::R8G8B8A8_SRGB);

#[cfg(feature = "egui")]
impl_format!(egui::Pos2, 1, vk::Format::R32G32_SFLOAT);
#[cfg(feature = "egui")]
impl_format!(egui::Color32, 1, vk::Format::R8G8B8A8_UNORM);

pub trait IndexInput: Copy {
    const VK_INDEX_TYPE: vk::IndexType;
}

impl IndexInput for u8 {
    const VK_INDEX_TYPE: vk::IndexType = vk::IndexType::UINT8_EXT;
}

impl IndexInput for u16 {
    const VK_INDEX_TYPE: vk::IndexType = vk::IndexType::UINT16;
}

impl IndexInput for u32 {
    const VK_INDEX_TYPE: vk::IndexType = vk::IndexType::UINT32;
}

pub trait VertexInput: Copy {
    fn binding_desc(binding: u32) -> vk::VertexInputBindingDescription;
    fn attr_desc(binding: u32) -> Vec<vk::VertexInputAttributeDescription>;
}

macro_rules! impl_vertex {
    (tuple : $($name:ident $idx:tt),+) => {
        impl<$($name: TypeFormat),+> VertexInput for ($($name,)+) {
            impl_vertex!(@binding_desc);

            #[allow(unused_assignments)]
            fn attr_desc(binding: u32) -> Vec<vk::VertexInputAttributeDescription> {
                let mut i = 0;
                vec![$({
                    let location = i;
                    i += $name::VK_WIDTH;
                    vk::VertexInputAttributeDescription {
                        binding,
                        location,
                        format: $name::VK_FORMAT,
                        offset: offset_of_tuple!(Self, $idx) as _,
                    }
                }),+]
            }
        }
    };

    (struct $name:ty : $($field:ident),+) => {
        impl VertexInput for $name {
            impl_vertex!(@binding_desc);

            #[allow(unused_assignments)]
            fn attr_desc(binding: u32) -> Vec<vk::VertexInputAttributeDescription> {
                let mut i = 0;
                vec![$({
                    let (format, width) = Self::lens_format(|v| &v.$field);
                    let location = i;
                    i += width;
                    vk::VertexInputAttributeDescription {
                        binding,
                        location,
                        format,
                        offset: offset_of!(Self, $field) as _,
                    }
                }),+]
            }
        }
    };

    (@binding_desc) => {
        fn binding_desc(binding: u32) -> vk::VertexInputBindingDescription {
            vk::VertexInputBindingDescription {
                binding,
                stride: std::mem::size_of::<Self>() as _,
                input_rate: vk::VertexInputRate::VERTEX,
            }
        }
    };
}

impl_vertex!(tuple: A 0);
impl_vertex!(tuple: A 0, B 1);
impl_vertex!(tuple: A 0, B 1, C 2);
impl_vertex!(tuple: A 0, B 1, C 2, D 3);
impl_vertex!(tuple: A 0, B 1, C 2, D 3, E 4);
impl_vertex!(tuple: A 0, B 1, C 2, D 3, E 4, F 5);

#[cfg(feature = "egui")]
impl_vertex!(struct egui::epaint::Vertex: pos, uv, color);

impl_vertex!(struct gltf_import::Vertex: position, normal, texcoord);

trait LensFormat {
    #[inline(always)]
    fn lens_format<T: TypeFormat, F: for<'a> FnOnce(&'a Self) -> &'a T>(_: F) -> (vk::Format, u32) {
        (T::VK_FORMAT, T::VK_WIDTH)
    }
}

impl<T> LensFormat for T {}
