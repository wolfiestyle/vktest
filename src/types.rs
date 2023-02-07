use ash::vk;

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