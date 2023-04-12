use crate::types::{ErrorDescription, VulkanResult};
use ash::vk;

pub trait CreateFromInfo {
    type Output;

    fn create(&self, device: &ash::Device) -> VulkanResult<Self::Output>;
}

impl CreateFromInfo for vk::ShaderModuleCreateInfo {
    type Output = vk::ShaderModule;

    #[inline]
    fn create(&self, device: &ash::Device) -> VulkanResult<Self::Output> {
        unsafe {
            device
                .create_shader_module(self, None)
                .describe_err("Failed to create shader module")
        }
    }
}

impl CreateFromInfo for vk::DescriptorSetLayoutCreateInfo {
    type Output = vk::DescriptorSetLayout;

    #[inline]
    fn create(&self, device: &ash::Device) -> VulkanResult<Self::Output> {
        unsafe {
            device
                .create_descriptor_set_layout(self, None)
                .describe_err("Failed to create descriptor set layout")
        }
    }
}

impl CreateFromInfo for vk::PipelineLayoutCreateInfo {
    type Output = vk::PipelineLayout;

    #[inline]
    fn create(&self, device: &ash::Device) -> VulkanResult<Self::Output> {
        unsafe {
            device
                .create_pipeline_layout(self, None)
                .describe_err("Failed to create pipeline layout")
        }
    }
}

impl CreateFromInfo for vk::PipelineCacheCreateInfo {
    type Output = vk::PipelineCache;

    #[inline]
    fn create(&self, device: &ash::Device) -> VulkanResult<Self::Output> {
        unsafe {
            device
                .create_pipeline_cache(self, None)
                .describe_err("Failed to create pipeline cache")
        }
    }
}

impl CreateFromInfo for vk::CommandPoolCreateInfo {
    type Output = vk::CommandPool;

    #[inline]
    fn create(&self, device: &ash::Device) -> VulkanResult<Self::Output> {
        unsafe { device.create_command_pool(self, None).describe_err("Failed to create command pool") }
    }
}

impl CreateFromInfo for vk::CommandBufferAllocateInfo {
    type Output = Vec<vk::CommandBuffer>;

    #[inline]
    fn create(&self, device: &ash::Device) -> VulkanResult<Self::Output> {
        unsafe {
            device
                .allocate_command_buffers(self)
                .describe_err("Failed to allocate command buffers")
        }
    }
}

impl CreateFromInfo for vk::SemaphoreCreateInfo {
    type Output = vk::Semaphore;

    #[inline]
    fn create(&self, device: &ash::Device) -> VulkanResult<Self::Output> {
        unsafe { device.create_semaphore(self, None).describe_err("Failed to create semaphore") }
    }
}

impl CreateFromInfo for vk::FenceCreateInfo {
    type Output = vk::Fence;

    #[inline]
    fn create(&self, device: &ash::Device) -> VulkanResult<Self::Output> {
        unsafe { device.create_fence(self, None).describe_err("Failed to create fence") }
    }
}

impl CreateFromInfo for vk::ImageViewCreateInfo {
    type Output = vk::ImageView;

    #[inline]
    fn create(&self, device: &ash::Device) -> VulkanResult<Self::Output> {
        unsafe { device.create_image_view(self, None).describe_err("Failed to create image view") }
    }
}

impl CreateFromInfo for vk::SamplerCreateInfo {
    type Output = vk::Sampler;

    #[inline]
    fn create(&self, device: &ash::Device) -> VulkanResult<Self::Output> {
        unsafe { device.create_sampler(self, None).describe_err("Failed to create texture sampler") }
    }
}

impl CreateFromInfo for vk::QueryPoolCreateInfo {
    type Output = vk::QueryPool;

    #[inline]
    fn create(&self, device: &ash::Device) -> VulkanResult<Self::Output> {
        unsafe { device.create_query_pool(self, None).describe_err("Failed to create query pool") }
    }
}
