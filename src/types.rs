use ash::vk;
use memoffset::offset_of_tuple;

pub type VulkanResult<T> = Result<T, VkError>;

#[derive(Debug)]
pub enum VkError {
    LoadingFailed(ash::LoadingError),
    Vulkan(vk::Result),
    VulkanMsg(&'static str, vk::Result),
    EngineError(&'static str),
    UnsuitableDevice, // used internally
}

impl std::fmt::Display for VkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Self::LoadingFailed(err) => write!(f, "Failed to load Vulkan library: {err}"),
            Self::Vulkan(err) => write!(f, "Vulkan error: {err}"),
            Self::VulkanMsg(msg, err) => write!(f, "{msg}: {err}"),
            Self::EngineError(desc) => write!(f, "{desc}"),
            Self::UnsuitableDevice => write!(f, "Unsuitable device"),
        }
    }
}

impl std::error::Error for VkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VkError::LoadingFailed(err) => Some(err),
            VkError::Vulkan(err) | VkError::VulkanMsg(_, err) => Some(err),
            _ => None,
        }
    }
}

impl From<ash::LoadingError> for VkError {
    fn from(err: ash::LoadingError) -> Self {
        VkError::LoadingFailed(err)
    }
}

impl From<vk::Result> for VkError {
    fn from(err: vk::Result) -> Self {
        VkError::Vulkan(err)
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

impl<T> VertexBindindDesc for T {
    fn binding_desc(binding: u32) -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(binding)
            .stride(std::mem::size_of::<Self>() as _)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
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
                    vk::VertexInputAttributeDescription::builder()
                        .binding(binding)
                        .location($idx)
                        .format($name::VK_FORMAT)
                        .offset(offset_of_tuple!(Self, $idx) as _)
                        .build()
                ),+]
            }
        }
    };
}

impl_tuple_desc!(A 0);
impl_tuple_desc!(A 0, B 1);
impl_tuple_desc!(A 0, B 1, C 2);
