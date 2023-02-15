use crate::instance::{DeviceInfo, DeviceSelection, SurfaceInfo, VulkanInstance};
use crate::types::*;
use ash::extensions::khr;
use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::array;
use std::mem::ManuallyDrop;
use std::sync::Arc;

pub struct VulkanDevice {
    instance: Arc<VulkanInstance>,
    surface: vk::SurfaceKHR,
    pub dev_info: DeviceInfo,
    pub surface_info: SurfaceInfo,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub graphics_pool: vk::CommandPool,
    pub swapchain_utils: khr::Swapchain,
    allocator: ga::GpuAllocator<vk::DeviceMemory>,
}

impl VulkanDevice {
    pub fn new<W>(window: &W, instance: Arc<VulkanInstance>, selection: DeviceSelection) -> VulkanResult<Self>
    where
        W: HasRawDisplayHandle + HasRawWindowHandle,
    {
        let surface = instance.create_surface(window)?;
        let dev_info = instance.pick_physical_device(surface, selection)?;
        eprintln!("Selected device: {:?}", dev_info.name);
        let surface_info = instance.query_surface_info(dev_info.phys_dev, surface)?;
        let device = instance.create_logical_device(&dev_info)?;
        let graphics_queue = unsafe { device.get_device_queue(dev_info.graphics_idx, 0) };
        let present_queue = unsafe { device.get_device_queue(dev_info.present_idx, 0) };
        let graphics_pool = Self::create_command_pool(&device, dev_info.graphics_idx, vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)?;
        let swapchain_utils = khr::Swapchain::new(&instance, &device);

        let alloc_config = ga::Config::i_am_prototyping();
        let dev_props = unsafe { gpu_alloc_ash::device_properties(&instance, 0, dev_info.phys_dev)? };
        let allocator = ga::GpuAllocator::new(alloc_config, dev_props);

        Ok(Self {
            instance,
            surface,
            dev_info,
            surface_info,
            device,
            graphics_queue,
            present_queue,
            graphics_pool,
            swapchain_utils,
            allocator,
        })
    }

    pub fn create_swapchain(
        &self, win_size: WinSize, image_count: u32, old_swapchain: Option<vk::SwapchainKHR>,
    ) -> VulkanResult<SwapchainInfo> {
        let surface_format = self.surface_info.surface_format();
        let present_mode = self.surface_info.present_mode();
        eprintln!("window size: {} x {}", win_size.width, win_size.height);
        let extent = self.surface_info.calc_extent(win_size.width, win_size.height);
        let img_count = self.surface_info.calc_image_count(image_count);
        let que_families = self.dev_info.unique_families();
        let img_sharing_mode = if que_families.len() > 1 {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let swapchain_ci = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface)
            .min_image_count(img_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(img_sharing_mode)
            .queue_family_indices(&que_families)
            .pre_transform(self.surface_info.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain.unwrap_or_default());

        let swapchain = unsafe {
            self.swapchain_utils
                .create_swapchain(&swapchain_ci, None)
                .describe_err("Failed to create swapchain")?
        };

        let images = unsafe {
            self.swapchain_utils
                .get_swapchain_images(swapchain)
                .describe_err("Failed to get swapchain images")?
        };

        let image_views = images
            .iter()
            .map(|&image| self.create_image_view(image, surface_format.format, vk::ImageAspectFlags::COLOR))
            .collect::<Result<_, _>>()?;

        Ok(SwapchainInfo {
            handle: swapchain,
            //images,
            image_views,
            format: surface_format.format,
            extent,
        })
    }

    pub fn create_shader_module(&self, spirv_code: &[u32]) -> VulkanResult<vk::ShaderModule> {
        let shader_ci = vk::ShaderModuleCreateInfo::builder().code(spirv_code);

        unsafe {
            self.device
                .create_shader_module(&shader_ci, None)
                .describe_err("Failed to create shader module")
        }
    }

    pub fn create_framebuffers(
        &self, swapchain: &SwapchainInfo, render_pass: vk::RenderPass, depth_imgviews: &[vk::ImageView],
    ) -> VulkanResult<Vec<vk::Framebuffer>> {
        assert_eq!(swapchain.image_views.len(), depth_imgviews.len());
        swapchain
            .image_views
            .iter()
            .zip(depth_imgviews)
            .map(|(&imgview, &depth_imgview)| {
                let attachments = [imgview, depth_imgview];
                let framebuffer_ci = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .layers(1);

                unsafe {
                    self.device
                        .create_framebuffer(&framebuffer_ci, None)
                        .describe_err("Failed to create framebuffer")
                }
            })
            .collect()
    }

    fn create_command_pool(device: &ash::Device, family_idx: u32, flags: vk::CommandPoolCreateFlags) -> VulkanResult<vk::CommandPool> {
        let command_pool_ci = vk::CommandPoolCreateInfo::builder().flags(flags).queue_family_index(family_idx);

        unsafe {
            device
                .create_command_pool(&command_pool_ci, None)
                .describe_err("Failed to create command pool")
        }
    }

    pub fn create_command_buffers(&self, count: u32) -> VulkanResult<Vec<vk::CommandBuffer>> {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.graphics_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .describe_err("Failed to allocate command buffers")
        }
    }

    pub fn begin_one_time_commands(&self) -> VulkanResult<vk::CommandBuffer> {
        let cmd_buffer = self.create_command_buffers(1)?[0];
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .describe_err("Failed to begin recording command buffer")?;
        }
        Ok(cmd_buffer)
    }

    pub fn end_one_time_commands(&self, cmd_buffer: vk::CommandBuffer, queue: vk::Queue) -> VulkanResult<()> {
        let submit_info = vk::SubmitInfo::builder().command_buffers(array::from_ref(&cmd_buffer)).build();
        unsafe {
            self.device
                .end_command_buffer(cmd_buffer)
                .describe_err("Failed to end recording command buffer")?;
            self.device
                .queue_submit(queue, array::from_ref(&submit_info), vk::Fence::null())
                .describe_err("Failed to submit queue")?;
            self.device.queue_wait_idle(queue).describe_err("Failed to wait queue idle")?;
            self.device.free_command_buffers(self.graphics_pool, array::from_ref(&cmd_buffer));
        }
        Ok(())
    }

    pub fn create_semaphore(&self) -> VulkanResult<vk::Semaphore> {
        let semaphore_ci = vk::SemaphoreCreateInfo::default();
        unsafe {
            self.device
                .create_semaphore(&semaphore_ci, None)
                .describe_err("Failed to create semaphore")
        }
    }

    pub fn create_fence(&self) -> VulkanResult<vk::Fence> {
        let fence_ci = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        unsafe { self.device.create_fence(&fence_ci, None).describe_err("Failed to create fence") }
    }

    pub fn update_surface_info(&mut self) -> VulkanResult<()> {
        self.surface_info = self.instance.query_surface_info(self.dev_info.phys_dev, self.surface)?;
        Ok(())
    }

    pub fn allocate_buffer(
        &mut self, size: vk::DeviceSize, buf_usage: vk::BufferUsageFlags, mem_usage: ga::UsageFlags,
    ) -> VulkanResult<VkBuffer> {
        let buffer_ci = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(buf_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_ci, None)
                .describe_err("Failed to create buffer")?
        };

        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory = unsafe {
            self.allocator.alloc(
                AshMemoryDevice::wrap(&self.device),
                ga::Request {
                    size: mem_reqs.size,
                    align_mask: mem_reqs.alignment,
                    usage: mem_usage,
                    memory_types: mem_reqs.memory_type_bits,
                },
            )?
        };
        //eprintln!("buffer alloc: usage = {buf_usage:?}\n{memory:#?}");

        unsafe {
            self.device
                .bind_buffer_memory(buffer, *memory.memory(), memory.offset())
                .describe_err("Failed to bind buffer memory")?
        };

        Ok(VkBuffer::new(buffer, memory))
    }

    fn write_memory_slice<T: Copy>(&self, memory: &mut MemoryBlock, data: &[T]) -> VulkanResult<()> {
        unsafe {
            let (prefix, bytes, suffix) = data.align_to();
            assert!(prefix.is_empty() && suffix.is_empty());
            memory.write_bytes(AshMemoryDevice::wrap(&self.device), 0, bytes)?;
        }
        Ok(())
    }

    fn copy_buffer(&self, dst_buffer: vk::Buffer, src_buffer: vk::Buffer, size: vk::DeviceSize) -> VulkanResult<()> {
        let cmd_buffer = self.begin_one_time_commands()?;
        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        unsafe {
            self.device.cmd_copy_buffer(cmd_buffer, src_buffer, dst_buffer, &[copy_region]);
        }
        self.end_one_time_commands(cmd_buffer, self.graphics_queue)
    }

    pub fn create_buffer<T: Copy>(&mut self, data: &[T], usage: vk::BufferUsageFlags) -> VulkanResult<VkBuffer> {
        let size = std::mem::size_of_val(data) as _;
        let mut src_buffer = self.allocate_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            ga::UsageFlags::UPLOAD | ga::UsageFlags::TRANSIENT,
        )?;
        let dst_buffer = self.allocate_buffer(size, vk::BufferUsageFlags::TRANSFER_DST | usage, ga::UsageFlags::FAST_DEVICE_ACCESS)?;

        self.write_memory_slice(&mut src_buffer.memory, data)?;
        self.copy_buffer(*dst_buffer, *src_buffer, size)?;

        unsafe { src_buffer.cleanup(self) };

        Ok(dst_buffer)
    }

    pub fn create_uniform_buffer<T>(&mut self) -> VulkanResult<UniformBuffer<T>> {
        let size = std::mem::size_of::<T>();
        let mut buffer = self.allocate_buffer(size as _, vk::BufferUsageFlags::UNIFORM_BUFFER, ga::UsageFlags::UPLOAD)?;
        let ub_mapped = unsafe { buffer.memory.map(AshMemoryDevice::wrap(self), 0, size)?.cast() };
        Ok(UniformBuffer {
            buffer,
            ub_mapped: Some(ub_mapped),
        })
    }

    pub fn allocate_image(
        &mut self, width: u32, height: u32, format: vk::Format, tiling: vk::ImageTiling, img_usage: vk::ImageUsageFlags,
        mem_usage: ga::UsageFlags,
    ) -> VulkanResult<VkImage> {
        let image_ci = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(img_usage)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe { self.device.create_image(&image_ci, None).describe_err("Failed to create image")? };

        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };
        let memory = unsafe {
            self.allocator.alloc(
                AshMemoryDevice::wrap(&self.device),
                ga::Request {
                    size: mem_reqs.size,
                    align_mask: mem_reqs.alignment,
                    usage: mem_usage,
                    memory_types: mem_reqs.memory_type_bits,
                },
            )?
        };
        //eprintln!("image alloc: usage = {img_usage:?}\n{memory:#?}");

        unsafe {
            self.device
                .bind_image_memory(image, *memory.memory(), memory.offset())
                .describe_err("Failed to bind image memory")?;
        }

        Ok(VkImage::new(image, memory))
    }

    fn transition_image_layout(&self, image: vk::Image, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout) -> VulkanResult<()> {
        let (src_access, dst_access, src_stage, dst_stage) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => return Err(VkError::EngineError("Unsupported layout transition")),
        };
        let cmd_buffer = self.begin_one_time_commands()?;
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .build();
        unsafe {
            self.device
                .cmd_pipeline_barrier(cmd_buffer, src_stage, dst_stage, Default::default(), &[], &[], &[barrier]);
        }
        self.end_one_time_commands(cmd_buffer, self.graphics_queue)?;
        Ok(())
    }

    fn copy_buffer_to_image(&self, buffer: vk::Buffer, image: vk::Image, width: u32, height: u32) -> VulkanResult<()> {
        let cmd_buffer = self.begin_one_time_commands()?;
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(Default::default())
            .image_extent(vk::Extent3D { width, height, depth: 1 })
            .build();
        unsafe {
            self.device
                .cmd_copy_buffer_to_image(cmd_buffer, buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
        }
        self.end_one_time_commands(cmd_buffer, self.graphics_queue)
    }

    pub fn create_texture(&mut self, width: u32, height: u32, data: &[u8]) -> VulkanResult<VkImage> {
        let size = width as vk::DeviceSize * height as vk::DeviceSize * 4;
        if data.len() != size as usize {
            return Err(VkError::EngineError("Image size and data length doesn't match"));
        }
        let mut src_buffer = self.allocate_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            ga::UsageFlags::UPLOAD | ga::UsageFlags::TRANSIENT,
        )?;
        let tex_image = self.allocate_image(
            width,
            height,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            ga::UsageFlags::FAST_DEVICE_ACCESS,
        )?;

        self.write_memory_slice(&mut src_buffer.memory, data)?;
        self.transition_image_layout(*tex_image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
        self.copy_buffer_to_image(*src_buffer, *tex_image, width, height)?;
        self.transition_image_layout(
            *tex_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;

        unsafe { src_buffer.cleanup(self) };

        Ok(tex_image)
    }

    pub fn create_image_view(
        &self, image: vk::Image, format: vk::Format, aspect_mask: vk::ImageAspectFlags,
    ) -> VulkanResult<vk::ImageView> {
        let imageview_ci = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            self.device
                .create_image_view(&imageview_ci, None)
                .describe_err("Failed to create image view")
        }
    }

    pub fn create_texture_sampler(&self, addr_mode: vk::SamplerAddressMode) -> VulkanResult<vk::Sampler> {
        let sampler_ci = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(addr_mode)
            .address_mode_v(addr_mode)
            .address_mode_w(addr_mode)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST);
        unsafe {
            self.device
                .create_sampler(&sampler_ci, None)
                .describe_err("Failed to create texture sampler")
        }
    }

    pub fn find_supported_format(
        &self, candidates: &[vk::Format], tiling: vk::ImageTiling, features: vk::FormatFeatureFlags,
    ) -> VulkanResult<vk::Format> {
        candidates
            .iter()
            .cloned()
            .find(|&fmt| {
                let props = unsafe { self.instance.get_physical_device_format_properties(self.dev_info.phys_dev, fmt) };
                (tiling == vk::ImageTiling::LINEAR && props.linear_tiling_features.contains(features))
                    || (tiling == vk::ImageTiling::OPTIMAL && props.optimal_tiling_features.contains(features))
            })
            .ok_or(VkError::EngineError("Failed to find supported image format"))
    }

    pub fn find_depth_format(&self) -> VulkanResult<vk::Format> {
        let formats = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
            vk::Format::D16_UNORM,
            vk::Format::D16_UNORM_S8_UINT,
        ];
        self.find_supported_format(&formats, vk::ImageTiling::OPTIMAL, vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
    }
}

impl std::ops::Deref for VulkanDevice {
    type Target = ash::Device;
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.allocator.cleanup(AshMemoryDevice::wrap(&self.device));
            self.device.destroy_command_pool(self.graphics_pool, None);
            self.instance.surface_utils.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
        }
    }
}

#[derive(Debug)]
pub struct SwapchainInfo {
    pub handle: vk::SwapchainKHR,
    //images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl SwapchainInfo {
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

    pub fn extent_rect(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: Default::default(),
            extent: self.extent,
        }
    }
}

impl std::ops::Deref for SwapchainInfo {
    type Target = vk::SwapchainKHR;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Cleanup<VulkanDevice> for SwapchainInfo {
    unsafe fn cleanup(&mut self, device: &mut VulkanDevice) {
        for &imgview in &self.image_views {
            device.destroy_image_view(imgview, None);
        }
        device.swapchain_utils.destroy_swapchain(self.handle, None);
    }
}

pub type MemoryBlock = ga::MemoryBlock<vk::DeviceMemory>;
pub type VkBuffer = MemoryObject<vk::Buffer>;
pub type VkImage = MemoryObject<vk::Image>;

#[derive(Debug)]
pub struct MemoryObject<T> {
    pub handle: T,
    pub memory: ManuallyDrop<MemoryBlock>,
}

impl<T> MemoryObject<T> {
    fn new(handle: T, memory: MemoryBlock) -> Self {
        Self {
            handle,
            memory: ManuallyDrop::new(memory),
        }
    }

    unsafe fn free_memory(&mut self, device: &mut VulkanDevice) {
        device
            .allocator
            .dealloc(AshMemoryDevice::wrap(&device.device), ManuallyDrop::take(&mut self.memory));
    }
}

impl<T> std::ops::Deref for MemoryObject<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Cleanup<VulkanDevice> for MemoryObject<vk::Buffer> {
    unsafe fn cleanup(&mut self, device: &mut VulkanDevice) {
        device.destroy_buffer(self.handle, None);
        self.free_memory(device);
    }
}

impl Cleanup<VulkanDevice> for MemoryObject<vk::Image> {
    unsafe fn cleanup(&mut self, device: &mut VulkanDevice) {
        device.destroy_image(self.handle, None);
        self.free_memory(device);
    }
}

pub struct UniformBuffer<T> {
    pub buffer: VkBuffer,
    ub_mapped: Option<std::ptr::NonNull<T>>,
}

impl<T> UniformBuffer<T> {
    pub fn write_uniforms(&mut self, ubo: T) {
        unsafe {
            std::ptr::write_volatile(self.ub_mapped.expect("use after free").as_ptr(), ubo);
        }
    }

    pub const fn buffer_size(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

impl<T> Cleanup<VulkanDevice> for UniformBuffer<T> {
    unsafe fn cleanup(&mut self, device: &mut VulkanDevice) {
        self.ub_mapped = None;
        self.buffer.memory.unmap(AshMemoryDevice::wrap(device));
        self.buffer.cleanup(device);
    }
}
