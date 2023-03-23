use crate::device::{ImageParams, VkImage, VulkanDevice};
use crate::types::*;
use ash::vk;
use glam::UVec2;
use gpu_allocator::MemoryLocation;

#[derive(Debug)]
pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub samples: vk::SampleCountFlags,
    pub msaa_image: Option<VkImage>,
    pub msaa_imgview: Option<vk::ImageView>,
    pub depth_image: Option<VkImage>,
    pub depth_imgview: Option<vk::ImageView>,
    pub depth_format: vk::Format,
}

impl Swapchain {
    pub const SUBRESOURCE_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    pub fn new(
        device: &VulkanDevice, win_size: UVec2, image_count: u32, depth_format: vk::Format, samples: vk::SampleCountFlags,
    ) -> VulkanResult<Self> {
        let mut swapchain = Swapchain::create(device, win_size.x, win_size.y, image_count, vk::SwapchainKHR::null())?;
        swapchain.create_msaa_attachment(device, samples)?;
        swapchain.create_depth_attachment(device, depth_format)?;
        Ok(swapchain)
    }

    pub fn recreate(&mut self, device: &VulkanDevice, win_size: UVec2) -> VulkanResult<()> {
        let mut swapchain = Swapchain::create(device, win_size.x, win_size.y, self.images.len() as _, self.handle)?;
        swapchain.create_msaa_attachment(device, self.samples)?;
        swapchain.create_depth_attachment(device, self.depth_format)?;
        let mut old_swapchain = std::mem::replace(self, swapchain);
        unsafe {
            device.device_wait_idle()?;
            old_swapchain.cleanup(device);
        }
        Ok(())
    }

    fn create(device: &VulkanDevice, width: u32, height: u32, image_count: u32, old_swapchain: vk::SwapchainKHR) -> VulkanResult<Self> {
        let surface_info = device.instance.query_surface_info(device.dev_info.phys_dev, device.surface)?;
        let surface_format = surface_info.find_surface_format();
        let present_mode = surface_info.find_present_mode(vk::PresentModeKHR::IMMEDIATE);
        let extent = surface_info.calc_extent(width, height);
        let img_count = surface_info.calc_image_count(image_count);
        let img_sharing_mode = if device.dev_info.unique_families.len() > 1 {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let swapchain_ci = vk::SwapchainCreateInfoKHR::builder()
            .surface(device.surface)
            .min_image_count(img_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(img_sharing_mode)
            .queue_family_indices(&device.dev_info.unique_families)
            .pre_transform(surface_info.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        let swapchain = unsafe {
            device
                .swapchain_fn
                .create_swapchain(&swapchain_ci, None)
                .describe_err("Failed to create swapchain")?
        };

        let images = unsafe {
            device
                .swapchain_fn
                .get_swapchain_images(swapchain)
                .describe_err("Failed to get swapchain images")?
        };

        let image_views = images
            .iter()
            .map(|&image| {
                vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .format(surface_format.format)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .subresource_range(Self::SUBRESOURCE_RANGE)
                    .create(device)
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            handle: swapchain,
            images,
            image_views,
            format: surface_format.format,
            extent,
            samples: vk::SampleCountFlags::TYPE_1,
            msaa_image: None,
            msaa_imgview: None,
            depth_image: None,
            depth_imgview: None,
            depth_format: vk::Format::UNDEFINED,
        })
    }

    fn create_msaa_attachment(&mut self, device: &VulkanDevice, samples: vk::SampleCountFlags) -> VulkanResult<()> {
        if samples == vk::SampleCountFlags::TYPE_1 {
            return Ok(());
        }
        let image = device.allocate_image(
            ImageParams {
                width: self.extent.width,
                height: self.extent.height,
                samples,
                format: self.format,
                ..Default::default()
            },
            vk::ImageCreateFlags::empty(),
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            MemoryLocation::GpuOnly,
            "MSAA image",
        )?;
        let imgview = vk::ImageViewCreateInfo::builder()
            .image(*image)
            .format(self.format)
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(image.props.subresource_range())
            .create(device)?;

        device.debug(|d| {
            d.set_object_name(device, &*image, "MSAA color image");
            d.set_object_name(device, &imgview, "MSAA color image view");
        });
        self.samples = samples;
        self.msaa_image = Some(image);
        self.msaa_imgview = Some(imgview);
        Ok(())
    }

    fn create_depth_attachment(&mut self, device: &VulkanDevice, depth_format: vk::Format) -> VulkanResult<()> {
        if depth_format == vk::Format::UNDEFINED {
            return VkError::EngineError("depth format undefined").into();
        }
        let depth_image = device.allocate_image(
            ImageParams {
                width: self.extent.width,
                height: self.extent.height,
                samples: self.samples,
                format: depth_format,
                ..Default::default()
            },
            vk::ImageCreateFlags::empty(),
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            MemoryLocation::GpuOnly,
            "Depth image",
        )?;

        let depth_imgview = vk::ImageViewCreateInfo::builder()
            .image(*depth_image)
            .format(depth_format)
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(depth_image.props.subresource_range())
            .create(device)?;

        device.debug(|d| {
            d.set_object_name(device, &*depth_image, "Depth image");
            d.set_object_name(device, &depth_imgview, "Depth image view");
        });

        self.depth_image = Some(depth_image);
        self.depth_imgview = Some(depth_imgview);
        self.depth_format = depth_format;
        Ok(())
    }

    #[inline]
    pub fn viewport(&self) -> vk::Viewport {
        vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.extent.width as f32,
            height: self.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    #[inline]
    pub fn extent_rect(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: Default::default(),
            extent: self.extent,
        }
    }

    #[inline]
    pub fn aspect(&self) -> f32 {
        self.extent.width as f32 / self.extent.height as f32
    }
}

impl std::ops::Deref for Swapchain {
    type Target = vk::SwapchainKHR;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Cleanup<VulkanDevice> for Swapchain {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        self.image_views.cleanup(device);
        self.msaa_imgview.cleanup(device);
        self.msaa_image.cleanup(device);
        self.depth_imgview.cleanup(device);
        self.depth_image.cleanup(device);
        device.swapchain_fn.destroy_swapchain(self.handle, None);
    }
}
