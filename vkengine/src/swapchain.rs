use crate::create::CreateFromInfo;
use crate::device::{ImageParams, VkImage, VulkanDevice};
use crate::types::{ErrorDescription, VkError, VulkanResult};
use ash::vk;
use glam::UVec2;
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

pub struct Swapchain {
    device: Arc<VulkanDevice>,
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
        device: &Arc<VulkanDevice>, win_size: UVec2, image_count: u32, depth_format: vk::Format, samples: vk::SampleCountFlags, vsync: bool,
    ) -> VulkanResult<Self> {
        let mut swapchain = Swapchain::create(device, win_size.x, win_size.y, image_count, vsync, vk::SwapchainKHR::null())?;
        swapchain.create_msaa_attachment(samples)?;
        swapchain.create_depth_attachment(depth_format)?;
        Ok(swapchain)
    }

    pub fn recreate(&mut self, win_size: UVec2) -> VulkanResult<()> {
        let mut swapchain = Swapchain::create(
            &self.device,
            win_size.x,
            win_size.y,
            self.images.len() as _,
            self.vsync,
            self.handle,
        )?;
        swapchain.create_msaa_attachment(self.samples)?;
        swapchain.create_depth_attachment(self.depth_format)?;
        let _old_swapchain = std::mem::replace(self, swapchain);
        unsafe { self.device.device_wait_idle()? };
        // old_swapchain is dropped here
        Ok(())
    }

    fn create(
        device: &Arc<VulkanDevice>, width: u32, height: u32, image_count: u32, vsync: bool, old_swapchain: vk::SwapchainKHR,
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

        let swapchain_ci = vk::SwapchainCreateInfoKHR::default()
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
                vk::ImageViewCreateInfo::default()
                    .image(image)
                    .format(surface_format.format)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .subresource_range(Self::SUBRESOURCE_RANGE)
                    .create(device)
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            device: Arc::clone(device),
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

    fn create_msaa_attachment(&mut self, samples: vk::SampleCountFlags) -> VulkanResult<()> {
        if samples == vk::SampleCountFlags::TYPE_1 {
            return Ok(());
        }
        let image = self.device.allocate_image(
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
        let imgview = image.create_view(&self.device, vk::ImageViewType::TYPE_2D)?;

        self.device.debug(|d| {
            d.set_object_name(*image, "MSAA color image");
            d.set_object_name(imgview, "MSAA color image view");
        });
        self.samples = samples;
        self.msaa_image = Some(image);
        self.msaa_imgview = Some(imgview);
        Ok(())
    }

    fn create_depth_attachment(&mut self, depth_format: vk::Format) -> VulkanResult<()> {
        if depth_format == vk::Format::UNDEFINED {
            return VkError::EngineError("depth format undefined").into();
        }
        let depth_image = self.device.allocate_image(
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

        let depth_imgview = depth_image.create_view(&self.device, vk::ImageViewType::TYPE_2D)?;

        self.device.debug(|d| {
            d.set_object_name(*depth_image, "Depth image");
            d.set_object_name(depth_imgview, "Depth image view");
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

    pub(crate) fn color_attachment(&self, image_idx: usize) -> vk::RenderingAttachmentInfo<'_> {
        if let Some(msaa_imgview) = self.msaa_imgview {
            vk::RenderingAttachmentInfo::default()
                .image_view(msaa_imgview)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                .resolve_image_view(self.image_views[image_idx])
                .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        } else {
            vk::RenderingAttachmentInfo::default()
                .image_view(self.image_views[image_idx])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
        }
    }

    pub(crate) fn depth_attachment(&self, rev_depth: bool) -> vk::RenderingAttachmentInfo<'_> {
        vk::RenderingAttachmentInfo::default()
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
    }
}

impl std::ops::Deref for Swapchain {
    type Target = vk::SwapchainKHR;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            for view in &self.image_views {
                self.device.destroy_image_view(*view, None);
            }
            if let Some(view) = self.msaa_imgview {
                self.device.destroy_image_view(view, None);
            }
            if let Some(view) = self.depth_imgview {
                self.device.destroy_image_view(view, None);
            }
            self.device.swapchain_fn.destroy_swapchain(self.handle, None);
        }
    }
}

impl std::fmt::Debug for Swapchain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Swapchain")
            .field("handle", &self.handle)
            .field("format", &self.format)
            .field("extent", &self.extent)
            .field("samples", &self.samples)
            .field("vsync", &self.vsync)
            .finish_non_exhaustive()
    }
}
