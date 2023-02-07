use ash::vk;

pub type VulkanResult<T> = Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    LoadingError(ash::LoadingError),
    VulkanError(&'static str, vk::Result),
    EngineError(&'static str),
    UnsuitableDevice, // used internally
}

impl Error {
    pub const fn bind_msg(msg: &'static str) -> impl Fn(vk::Result) -> Self {
        move |err| Self::VulkanError(msg, err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Self::LoadingError(err) => write!(f, "Failed to load Vulkan library: {err}"),
            Self::VulkanError(desc, err) => write!(f, "{desc}: {err}"),
            Self::EngineError(desc) => write!(f, "{desc}"),
            Self::UnsuitableDevice => write!(f, "Unsuitable device"),
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

impl<A: TypeFormat, B: TypeFormat> VertexAttrDesc for (A, B) {
    fn attr_desc(binding: u32) -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription::builder()
                .binding(binding)
                .location(0)
                .format(A::VK_FORMAT)
                .offset(0)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(binding)
                .location(1)
                .format(B::VK_FORMAT)
                .offset(std::mem::size_of::<A>() as _)
                .build(),
        ]
    }
}
