use crate::create::CreateFromInfo;
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
    pub vsync: bool,
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
        device: &VulkanDevice, win_size: UVec2, image_count: u32, depth_format: vk::Format, samples: vk::SampleCountFlags, vsync: bool,
    ) -> VulkanResult<Self> {
        let mut swapchain = Swapchain::create(device, win_size.x, win_size.y, image_count, vsync, vk::SwapchainKHR::null())?;
        swapchain.create_msaa_attachment(device, samples)?;
        swapchain.create_depth_attachment(device, depth_format)?;
        Ok(swapchain)
    }

    pub fn recreate(&mut self, device: &VulkanDevice, win_size: UVec2) -> VulkanResult<()> {
        let mut swapchain = Swapchain::create(device, win_size.x, win_size.y, self.images.len() as _, self.vsync, self.handle)?;
        swapchain.create_msaa_attachment(device, self.samples)?;
        swapchain.create_depth_attachment(device, self.depth_format)?;
        let old_swapchain = std::mem::replace(self, swapchain);
        unsafe { device.device_wait_idle()? };
        device.dispose_of(old_swapchain);
        Ok(())
    }

    fn create(
        device: &VulkanDevice, width: u32, height: u32, image_count: u32, vsync: bool, old_swapchain: vk::SwapchainKHR,
    ) -> VulkanResult<Self> {
        let surface_info = device.instance.query_surface_info(device.dev_info.phys_dev, device.surface)?;
        let surface_format = surface_info.find_surface_format();
        let present_mode = surface_info.find_present_mode(vsync);
        eprintln!("present mode: {present_mode:?}");
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
            vsync,
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
        )?;
        let imgview = image.create_view(device, vk::ImageViewType::TYPE_2D)?;

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
        )?;

        let depth_imgview = depth_image.create_view(device, vk::ImageViewType::TYPE_2D)?;

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
    pub fn viewport_inv(&self) -> vk::Viewport {
        let height = self.extent.height as f32;
        vk::Viewport {
            x: 0.0,
            y: height,
            width: self.extent.width as f32,
            height: -height,
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

    #[inline]
    pub fn has_float_depth(&self) -> bool {
        matches!(self.depth_format, vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT)
    }

    pub(crate) fn color_attachment(&self, image_idx: usize) -> vk::RenderingAttachmentInfo {
        if let Some(msaa_imgview) = self.msaa_imgview {
            vk::RenderingAttachmentInfo::builder()
                .image_view(msaa_imgview)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                .resolve_image_view(self.image_views[image_idx])
                .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .build()
        } else {
            vk::RenderingAttachmentInfo::builder()
                .image_view(self.image_views[image_idx])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .build()
        }
    }

    pub(crate) fn depth_attachment(&self, rev_depth: bool) -> vk::RenderingAttachmentInfo {
        vk::RenderingAttachmentInfo::builder()
            .image_view(self.depth_imgview.expect("missing depth image view"))
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: if rev_depth { 0.0 } else { 1.0 },
                    stencil: 0,
                },
            })
            .build()
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
