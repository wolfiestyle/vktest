use ash::vk;
use memoffset::{offset_of, offset_of_tuple};

pub type VulkanResult<T> = Result<T, VkError>;

#[derive(Debug)]
pub enum VkError {
    LoadingFailed(ash::LoadingError),
    Vulkan(vk::Result),
    VulkanMsg(&'static str, vk::Result),
    Io(std::io::Error),
    Image(&'static str),
    EngineError(&'static str),
    UnsuitableDevice, // used internally
}

impl std::fmt::Display for VkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Self::LoadingFailed(err) => write!(f, "Failed to load Vulkan library: {err}"),
            Self::Vulkan(err) => write!(f, "Vulkan error: {err}"),
            Self::VulkanMsg(msg, err) => write!(f, "{msg}: {err}"),
            Self::Io(err) => write!(f, "IO error: {err}"),
            Self::Image(msg) => write!(f, "{msg}"),
            Self::EngineError(desc) => write!(f, "{desc}"),
            Self::UnsuitableDevice => write!(f, "Unsuitable device"),
        }
    }
}

impl std::error::Error for VkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LoadingFailed(err) => Some(err),
            Self::Vulkan(err) | Self::VulkanMsg(_, err) => Some(err),
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<ash::LoadingError> for VkError {
    fn from(err: ash::LoadingError) -> Self {
        Self::LoadingFailed(err)
    }
}

impl From<vk::Result> for VkError {
    fn from(err: vk::Result) -> Self {
        Self::Vulkan(err)
    }
}

impl From<std::io::Error> for VkError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

pub trait ErrorDescription<M> {
    type Output;

    fn describe_err(self, msg: M) -> VulkanResult<Self::Output>;
}

impl<T> ErrorDescription<&'static str> for ash::prelude::VkResult<T> {
    type Output = T;

    fn describe_err(self, msg: &'static str) -> VulkanResult<Self::Output> {
        self.map_err(|err| VkError::VulkanMsg(msg, err))
    }
}

impl<T> ErrorDescription<&'static str> for Option<(stb::image::Info, stb::image::Data<T>)> {
    type Output = (stb::image::Info, stb::image::Data<T>);

    fn describe_err(self, msg: &'static str) -> VulkanResult<Self::Output> {
        self.ok_or_else(|| VkError::Image(msg))
    }
}

pub trait Cleanup<C> {
    unsafe fn cleanup(&mut self, context: &C);
}

impl<C, T: Cleanup<C>> Cleanup<C> for [T] {
    unsafe fn cleanup(&mut self, context: &C) {
        for item in self {
            item.cleanup(context);
        }
    }
}

pub trait TypeFormat {
    const VK_FORMAT: vk::Format;
}

macro_rules! impl_format {
    ($type:ty, $val:expr) => {
        impl TypeFormat for $type {
            const VK_FORMAT: vk::Format = $val;
        }
    };
}

impl_format!([f32; 1], vk::Format::R32_SFLOAT);
impl_format!([f32; 2], vk::Format::R32G32_SFLOAT);
impl_format!([f32; 3], vk::Format::R32G32B32_SFLOAT);
impl_format!([f32; 4], vk::Format::R32G32B32A32_SFLOAT);

pub trait VertexBindindDesc {
    fn binding_desc(binding: u32) -> vk::VertexInputBindingDescription;
}

impl<T: VertexAttrDesc> VertexBindindDesc for T {
    fn binding_desc(binding: u32) -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding,
            stride: std::mem::size_of::<Self>() as _,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }
}

pub trait VertexAttrDesc {
    fn attr_desc(binding: u32) -> Vec<vk::VertexInputAttributeDescription>;
}

macro_rules! impl_tuple_desc {
    ($($name:ident $idx:tt),+) => {
        impl<$($name: TypeFormat),+> VertexAttrDesc for ($($name,)+) {
            fn attr_desc(binding: u32) -> Vec<vk::VertexInputAttributeDescription> {
                vec![$(
                    vk::VertexInputAttributeDescription {
                        binding,
                        location: $idx,
                        format: $name::VK_FORMAT,
                        offset: offset_of_tuple!(Self, $idx) as _,
                    }
                ),+]
            }
        }
    };
}

impl_tuple_desc!(A 0);
impl_tuple_desc!(A 0, B 1);
impl_tuple_desc!(A 0, B 1, C 2);

trait LensFormat {
    fn lens_format<T: TypeFormat, F: for<'a> FnOnce(&'a Self) -> &'a T>(_: F) -> vk::Format {
        T::VK_FORMAT
    }
}

impl<T> LensFormat for T {}

impl VertexAttrDesc for obj::TexturedVertex {
    fn attr_desc(binding: u32) -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                binding,
                location: 0,
                format: Self::lens_format(|v| &v.position),
                offset: offset_of!(Self, position) as _,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 1,
                format: Self::lens_format(|v| &v.normal),
                offset: offset_of!(Self, normal) as _,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 2,
                format: Self::lens_format(|v| &v.texture),
                offset: offset_of!(Self, texture) as _,
            },
        ]
    }
}
