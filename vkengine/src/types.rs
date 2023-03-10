use ash::vk;
use memoffset::{offset_of, offset_of_tuple};
use std::sync::Arc;
#[cfg(feature = "winit")]
use winit::dpi::PhysicalSize;

pub type VulkanResult<T> = Result<T, VkError>;

#[derive(Debug)]
pub enum VkError {
    LoadingFailed(ash::LoadingError),
    Vulkan(vk::Result),
    VulkanMsg(&'static str, vk::Result),
    MemoryAlloc(gpu_allocator::AllocationError),
    EngineError(&'static str),
    UnsuitableDevice, // used internally
    UnfinishedJob,    // used internally
}

impl std::fmt::Display for VkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Self::LoadingFailed(err) => write!(f, "Failed to load Vulkan library: {err}"),
            Self::Vulkan(err) => write!(f, "Vulkan error: {err}"),
            Self::VulkanMsg(msg, err) => write!(f, "{msg}: {err}"),
            Self::MemoryAlloc(err) => write!(f, "Memory allocation error: {err}"),
            Self::EngineError(desc) => write!(f, "{desc}"),
            Self::UnsuitableDevice => write!(f, "Unsuitable device"),
            Self::UnfinishedJob => write!(f, "Unfinished job"),
        }
    }
}

impl std::error::Error for VkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LoadingFailed(err) => Some(err),
            Self::Vulkan(err) | Self::VulkanMsg(_, err) => Some(err),
            Self::MemoryAlloc(err) => Some(err),
            _ => None,
        }
    }
}

impl<T> From<VkError> for VulkanResult<T> {
    fn from(value: VkError) -> Self {
        Self::Err(value)
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

impl From<gpu_allocator::AllocationError> for VkError {
    fn from(err: gpu_allocator::AllocationError) -> Self {
        Self::MemoryAlloc(err)
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

impl<T> ErrorDescription<&'static str> for Option<T> {
    type Output = T;

    fn describe_err(self, msg: &'static str) -> VulkanResult<Self::Output> {
        self.ok_or_else(|| VkError::EngineError(msg))
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

impl<C, K, V: Cleanup<C>> Cleanup<C> for std::collections::HashMap<K, V> {
    unsafe fn cleanup(&mut self, context: &C) {
        for item in self.values_mut() {
            item.cleanup(context);
        }
    }
}

impl<C, T: Cleanup<C>> Cleanup<C> for Option<T> {
    unsafe fn cleanup(&mut self, context: &C) {
        if let Some(item) = self {
            item.cleanup(context);
        }
    }
}

impl<C, T: Cleanup<C>> Cleanup<C> for Arc<T> {
    unsafe fn cleanup(&mut self, context: &C) {
        if let Some(item) = Arc::get_mut(self) {
            item.cleanup(context);
        }
    }
}

impl<C, T: Cleanup<C> + Send> Cleanup<C> for thread_local::ThreadLocal<T> {
    unsafe fn cleanup(&mut self, context: &C) {
        for item in self {
            item.cleanup(context);
        }
    }
}

impl Cleanup<ash::Device> for vk::CommandPool {
    unsafe fn cleanup(&mut self, device: &ash::Device) {
        device.destroy_command_pool(*self, None);
    }
}

impl Cleanup<ash::Device> for vk::Sampler {
    unsafe fn cleanup(&mut self, device: &ash::Device) {
        device.destroy_sampler(*self, None);
    }
}

impl Cleanup<ash::Device> for vk::DescriptorSetLayout {
    unsafe fn cleanup(&mut self, device: &ash::Device) {
        device.destroy_descriptor_set_layout(*self, None);
    }
}

impl Cleanup<ash::Device> for vk::ImageView {
    unsafe fn cleanup(&mut self, device: &ash::Device) {
        device.destroy_image_view(*self, None);
    }
}

pub trait WindowSize {
    fn window_size(&self) -> [u32; 2];
}

#[cfg(feature = "winit")]
impl WindowSize for winit::window::Window {
    #[inline]
    fn window_size(&self) -> [u32; 2] {
        let PhysicalSize { width, height } = self.inner_size();
        [width, height]
    }
}

pub trait TypeFormat {
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
impl_format!(u32, 1, vk::Format::R32_UINT);
impl_format!([u32; 1], 1, vk::Format::R32_UINT);
impl_format!([u32; 2], 1, vk::Format::R32G32_UINT);
impl_format!([u32; 3], 1, vk::Format::R32G32B32_UINT);
impl_format!([u32; 4], 1, vk::Format::R32G32B32A32_UINT);
impl_format!(i32, 1, vk::Format::R32_SINT);
impl_format!([i32; 1], 1, vk::Format::R32_SINT);
impl_format!([i32; 2], 1, vk::Format::R32G32_SINT);
impl_format!([i32; 3], 1, vk::Format::R32G32B32_SINT);
impl_format!([i32; 4], 1, vk::Format::R32G32B32A32_SINT);
#[cfg(feature = "egui")]
impl_format!(egui::Pos2, 1, vk::Format::R32G32_SFLOAT);
#[cfg(feature = "egui")]
impl_format!(egui::Color32, 1, vk::Format::R8G8B8A8_UNORM);

pub trait VertexInput {
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

trait LensFormat {
    #[inline(always)]
    fn lens_format<T: TypeFormat, F: for<'a> FnOnce(&'a Self) -> &'a T>(_: F) -> (vk::Format, u32) {
        (T::VK_FORMAT, T::VK_WIDTH)
    }
}

impl<T> LensFormat for T {}

pub trait ObjectType {
    const VK_OBJECT_TYPE: vk::ObjectType;
}

macro_rules! impl_object_type {
    ($type:ty, $val:expr) => {
        impl ObjectType for $type {
            const VK_OBJECT_TYPE: vk::ObjectType = $val;
        }
    };
}

impl_object_type!(vk::Instance, vk::ObjectType::INSTANCE);
impl_object_type!(vk::PhysicalDevice, vk::ObjectType::PHYSICAL_DEVICE);
impl_object_type!(vk::Device, vk::ObjectType::DEVICE);
impl_object_type!(vk::Queue, vk::ObjectType::QUEUE);
impl_object_type!(vk::Semaphore, vk::ObjectType::SEMAPHORE);
impl_object_type!(vk::CommandBuffer, vk::ObjectType::COMMAND_BUFFER);
impl_object_type!(vk::Fence, vk::ObjectType::FENCE);
impl_object_type!(vk::DeviceMemory, vk::ObjectType::DEVICE_MEMORY);
impl_object_type!(vk::Buffer, vk::ObjectType::BUFFER);
impl_object_type!(vk::Image, vk::ObjectType::IMAGE);
impl_object_type!(vk::Event, vk::ObjectType::EVENT);
impl_object_type!(vk::QueryPool, vk::ObjectType::QUERY_POOL);
impl_object_type!(vk::BufferView, vk::ObjectType::BUFFER_VIEW);
impl_object_type!(vk::ImageView, vk::ObjectType::IMAGE_VIEW);
impl_object_type!(vk::ShaderModule, vk::ObjectType::SHADER_MODULE);
impl_object_type!(vk::PipelineCache, vk::ObjectType::PIPELINE_CACHE);
impl_object_type!(vk::PipelineLayout, vk::ObjectType::PIPELINE_LAYOUT);
impl_object_type!(vk::RenderPass, vk::ObjectType::RENDER_PASS);
impl_object_type!(vk::Pipeline, vk::ObjectType::PIPELINE);
impl_object_type!(vk::DescriptorSetLayout, vk::ObjectType::DESCRIPTOR_SET_LAYOUT);
impl_object_type!(vk::Sampler, vk::ObjectType::SAMPLER);
impl_object_type!(vk::DescriptorPool, vk::ObjectType::DESCRIPTOR_POOL);
impl_object_type!(vk::DescriptorSet, vk::ObjectType::DESCRIPTOR_SET);
impl_object_type!(vk::Framebuffer, vk::ObjectType::FRAMEBUFFER);
impl_object_type!(vk::CommandPool, vk::ObjectType::COMMAND_POOL);

//HACK: same layout as easy_gltf's Vertex but with Copy
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VertexTemp {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub tex_coords: [f32; 2],
}

impl_vertex!(struct VertexTemp: position, normal, tex_coords);
