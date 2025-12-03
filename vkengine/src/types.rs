use ash::vk;
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

    #[cfg(feature = "winit")]
    #[error("EventLoop error: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),

    #[error("Raw window handle error: {0}")]
    WindowHandle(#[from] raw_window_handle::HandleError),

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
