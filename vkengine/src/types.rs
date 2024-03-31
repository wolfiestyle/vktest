use ash::vk;
use std::sync::Arc;
use thiserror::Error;
#[cfg(feature = "winit")]
use winit::dpi::PhysicalSize;

pub type VulkanResult<T> = Result<T, VkError>;

impl<T> From<VkError> for VulkanResult<T> {
    fn from(value: VkError) -> Self {
        Self::Err(value)
    }
}

#[derive(Debug, Error)]
pub enum VkError {
    #[error("Failed to load Vulkan library: {0}")]
    LoadingFailed(#[from] ash::LoadingError),

    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),

    #[error("{0}: {1}")]
    VulkanMsg(&'static str, #[source] vk::Result),

    #[error("Memory allocation error: {0}")]
    MemoryAlloc(#[from] gpu_allocator::AllocationError),

    #[error("Invalid argument: {0}")]
    InvalidArgument(&'static str),

    #[error("{0}")]
    EngineError(&'static str),

    #[error("Unsuitable device")]
    UnsuitableDevice, // used internally

    #[error("Unfinished job")]
    UnfinishedJob, // used internally
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

impl<C, T: Cleanup<C>> Cleanup<C> for Vec<T> {
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
