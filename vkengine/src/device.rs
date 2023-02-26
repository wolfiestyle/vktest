use crate::debug::DebugUtils;
use crate::instance::{DeviceInfo, DeviceSelection, VulkanInstance};
use crate::types::*;
use ash::extensions::khr;
use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::mem::ManuallyDrop;
use std::slice;
use std::sync::{Arc, Mutex};

pub struct VulkanDevice {
    instance: Arc<VulkanInstance>,
    surface: vk::SurfaceKHR,
    pub dev_info: DeviceInfo,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    transfer_pool: vk::CommandPool,
    pub swapchain_fn: khr::Swapchain,
    pub dynrender_fn: khr::DynamicRendering,
    pub pushdesc_fn: khr::PushDescriptor,
    allocator: Mutex<ga::GpuAllocator<vk::DeviceMemory>>,
}

impl VulkanDevice {
    pub fn new<W>(window: &W, instance: Arc<VulkanInstance>, selection: DeviceSelection) -> VulkanResult<Self>
    where
        W: HasRawDisplayHandle + HasRawWindowHandle,
    {
        let surface = instance.create_surface(window)?;
        let dev_info = instance.pick_physical_device(surface, selection)?;
        eprintln!("Selected device: {:?}", dev_info.name);
        let device = instance.create_logical_device(&dev_info)?;
        let graphics_queue = unsafe { device.get_device_queue(dev_info.graphics_idx, 0) };
        let present_queue = unsafe { device.get_device_queue(dev_info.present_idx, 0) };
        let swapchain_fn = khr::Swapchain::new(&instance, &device);
        let dynrender_fn = khr::DynamicRendering::new(&instance, &device);
        let pushdesc_fn = khr::PushDescriptor::new(&instance, &device);

        let alloc_config = ga::Config::i_am_prototyping();
        let dev_props = unsafe { gpu_alloc_ash::device_properties(&instance, 0, dev_info.phys_dev)? };
        let allocator = Mutex::new(ga::GpuAllocator::new(alloc_config, dev_props));

        let family = dev_info.graphics_idx;

        let mut this = Self {
            instance,
            surface,
            dev_info,
            device,
            graphics_queue,
            present_queue,
            transfer_pool: vk::CommandPool::null(),
            swapchain_fn,
            dynrender_fn,
            pushdesc_fn,
            allocator,
        };

        this.transfer_pool = this.create_command_pool(family, vk::CommandPoolCreateFlags::TRANSIENT)?;

        Ok(this)
    }

    pub fn create_swapchain(&self, win_size: WinSize, image_count: u32, depth_format: vk::Format) -> VulkanResult<Swapchain> {
        let mut swapchain = Swapchain::new(self, win_size, image_count, vk::SwapchainKHR::null())?;
        swapchain.create_depth_attachments(self, depth_format, swapchain.images.len())?;
        Ok(swapchain)
    }

    pub fn recreate_swapchain(&self, win_size: WinSize, old_swapchain: &Swapchain) -> VulkanResult<Swapchain> {
        let mut swapchain = Swapchain::new(self, win_size, old_swapchain.images.len() as u32, old_swapchain.handle)?;
        swapchain.create_depth_attachments(self, old_swapchain.depth_format, swapchain.images.len())?;
        Ok(swapchain)
    }

    pub fn create_shader_module(&self, spirv_code: &[u32]) -> VulkanResult<vk::ShaderModule> {
        let shader_ci = vk::ShaderModuleCreateInfo::builder().code(spirv_code);

        unsafe {
            self.device
                .create_shader_module(&shader_ci, None)
                .describe_err("Failed to create shader module")
        }
    }

    pub fn create_descriptor_set_layout(
        &self, layout_bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> VulkanResult<vk::DescriptorSetLayout> {
        let desc_layout_ci = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(layout_bindings);

        unsafe {
            self.device
                .create_descriptor_set_layout(&desc_layout_ci, None)
                .describe_err("Failed to create descriptor set layout")
        }
    }

    pub fn create_pipeline_layout(
        &self, set_layouts: &[vk::DescriptorSetLayout], push_constants: &[vk::PushConstantRange],
    ) -> VulkanResult<vk::PipelineLayout> {
        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_constants);

        unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_ci, None)
                .describe_err("Failed to create pipeline layout")
        }
    }

    pub fn create_command_pool(&self, family_idx: u32, flags: vk::CommandPoolCreateFlags) -> VulkanResult<vk::CommandPool> {
        let command_pool_ci = vk::CommandPoolCreateInfo::builder().flags(flags).queue_family_index(family_idx);

        unsafe {
            self.device
                .create_command_pool(&command_pool_ci, None)
                .describe_err("Failed to create command pool")
        }
    }

    pub fn create_command_buffers(
        &self, pool: vk::CommandPool, count: u32, level: vk::CommandBufferLevel,
    ) -> VulkanResult<Vec<vk::CommandBuffer>> {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(level)
            .command_buffer_count(count);

        unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .describe_err("Failed to allocate command buffers")
        }
    }

    fn begin_one_time_commands(&self) -> VulkanResult<vk::CommandBuffer> {
        let cmd_buffer = self.create_command_buffers(self.transfer_pool, 1, vk::CommandBufferLevel::PRIMARY)?[0];
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .describe_err("Failed to begin recording command buffer")?;
        }
        Ok(cmd_buffer)
    }

    fn end_one_time_commands(&self, cmd_buffer: vk::CommandBuffer) -> VulkanResult<()> {
        let submit_info = vk::SubmitInfo::builder().command_buffers(slice::from_ref(&cmd_buffer));
        unsafe {
            self.device
                .end_command_buffer(cmd_buffer)
                .describe_err("Failed to end recording command buffer")?;
            self.device
                .queue_submit(self.graphics_queue, slice::from_ref(&submit_info), vk::Fence::null())
                .describe_err("Failed to submit queue")?;
            self.device
                .queue_wait_idle(self.graphics_queue)
                .describe_err("Failed to wait queue idle")?;
            self.device.free_command_buffers(self.transfer_pool, slice::from_ref(&cmd_buffer));
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

    pub fn allocate_buffer(
        &self, size: vk::DeviceSize, buf_usage: vk::BufferUsageFlags, mem_usage: ga::UsageFlags,
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
            self.allocator.lock().unwrap().alloc(
                AshMemoryDevice::wrap(&self.device),
                ga::Request {
                    size: mem_reqs.size,
                    align_mask: mem_reqs.alignment - 1,
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
        self.end_one_time_commands(cmd_buffer)
    }

    pub fn create_buffer_from_data<T: Copy>(&self, data: &[T], usage: vk::BufferUsageFlags) -> VulkanResult<VkBuffer> {
        let size = std::mem::size_of_val(data) as _;
        let mut src_buffer = self.allocate_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            ga::UsageFlags::UPLOAD | ga::UsageFlags::TRANSIENT,
        )?;
        let dst_buffer = self.allocate_buffer(size, vk::BufferUsageFlags::TRANSFER_DST | usage, ga::UsageFlags::FAST_DEVICE_ACCESS)?;

        src_buffer.write_slice(self, data, 0)?;
        self.copy_buffer(*dst_buffer, *src_buffer, size)?;

        unsafe { src_buffer.cleanup(self) };

        Ok(dst_buffer)
    }

    pub fn create_uniform_buffer<T>(&self) -> VulkanResult<UniformBuffer<T>> {
        let size = std::mem::size_of::<T>();
        let mut buffer = self.allocate_buffer(size as _, vk::BufferUsageFlags::UNIFORM_BUFFER, ga::UsageFlags::UPLOAD)?;
        let ub_mapped = unsafe { buffer.memory.map(AshMemoryDevice::wrap(self), 0, size)?.cast() };
        Ok(UniformBuffer {
            buffer,
            ub_mapped: Some(ub_mapped),
        })
    }

    pub fn allocate_image(
        &self, width: u32, height: u32, layers: u32, format: vk::Format, flags: vk::ImageCreateFlags, img_usage: vk::ImageUsageFlags,
        mem_usage: ga::UsageFlags,
    ) -> VulkanResult<VkImage> {
        let image_ci = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(layers)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(img_usage)
            .flags(flags)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe { self.device.create_image(&image_ci, None).describe_err("Failed to create image")? };

        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };
        let memory = unsafe {
            self.allocator.lock().unwrap().alloc(
                AshMemoryDevice::wrap(&self.device),
                ga::Request {
                    size: mem_reqs.size,
                    align_mask: mem_reqs.alignment - 1,
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

    pub fn transition_image_layout(
        &self, cmd_buffer: vk::CommandBuffer, image: vk::Image, format: vk::Format, layer_count: u32, old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
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
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            ),
            (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR) => (
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            ),
            _ => panic!("Unsupported layout transition"),
        };

        let aspect_mask = match format {
            vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }
            vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
            _ => vk::ImageAspectFlags::COLOR,
        };

        let barrier = vk::ImageMemoryBarrier::builder()
            .image(image)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count,
            });
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd_buffer,
                src_stage,
                dst_stage,
                Default::default(),
                &[],
                &[],
                slice::from_ref(&barrier),
            );
        }
    }

    fn copy_buffer_to_image(
        &self, cmd_buffer: vk::CommandBuffer, buffer: vk::Buffer, image: vk::Image, width: u32, height: u32, layer_count: u32,
    ) {
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count,
            })
            .image_offset(Default::default())
            .image_extent(vk::Extent3D { width, height, depth: 1 });
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                cmd_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                slice::from_ref(&region),
            );
        }
    }

    pub fn create_image_from_data(&self, width: u32, height: u32, data: ImageData) -> VulkanResult<VkImage> {
        let layer_size = width as usize * height as usize * 4;
        let layers = data.layer_count();
        let size = layer_size as vk::DeviceSize * layers as vk::DeviceSize;
        let mut src_buffer = self.allocate_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            ga::UsageFlags::UPLOAD | ga::UsageFlags::TRANSIENT,
        )?;
        let flags = if layers == 6 {
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            Default::default()
        };
        let tex_image = self.allocate_image(
            width,
            height,
            layers,
            vk::Format::R8G8B8A8_SRGB,
            flags,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            ga::UsageFlags::FAST_DEVICE_ACCESS,
        )?;

        match data {
            ImageData::Single(bytes) => {
                src_buffer.write_bytes(self, bytes, 0)?;
            }
            ImageData::Array(n, bytes) => {
                for i in 0..n as usize {
                    src_buffer.write_bytes(self, bytes[i], layer_size * i)?;
                }
            }
        }
        let cmd_buffer = self.begin_one_time_commands()?;
        self.transition_image_layout(
            cmd_buffer,
            *tex_image,
            vk::Format::R8G8B8A8_SRGB,
            layers,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        self.copy_buffer_to_image(cmd_buffer, *src_buffer, *tex_image, width, height, layers);
        self.transition_image_layout(
            cmd_buffer,
            *tex_image,
            vk::Format::R8G8B8A8_SRGB,
            layers,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        self.end_one_time_commands(cmd_buffer)?;

        unsafe { src_buffer.cleanup(self) };
        self.debug(|d| d.set_object_name(&self.device, &tex_image.handle, "Texture image"));

        Ok(tex_image)
    }

    pub fn create_image_view(
        &self, image: vk::Image, format: vk::Format, view_type: vk::ImageViewType, aspect_mask: vk::ImageAspectFlags,
    ) -> VulkanResult<vk::ImageView> {
        let layer_count = if view_type == vk::ImageViewType::CUBE { 6 } else { 1 };
        let imageview_ci = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(view_type)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count,
            });

        unsafe {
            self.device
                .create_image_view(&imageview_ci, None)
                .describe_err("Failed to create image view")
        }
    }

    pub fn create_texture_sampler(
        &self, mag_filter: vk::Filter, min_filter: vk::Filter, addr_mode: vk::SamplerAddressMode,
    ) -> VulkanResult<vk::Sampler> {
        let sampler_ci = vk::SamplerCreateInfo::builder()
            .mag_filter(mag_filter)
            .min_filter(min_filter)
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
                match tiling {
                    vk::ImageTiling::LINEAR => props.linear_tiling_features.contains(features),
                    vk::ImageTiling::OPTIMAL => props.optimal_tiling_features.contains(features),
                    _ => panic!("Unsupported tiling mode"),
                }
            })
            .ok_or(VkError::EngineError("Failed to find supported image format"))
    }

    pub fn find_depth_format(&self, stencil: bool) -> VulkanResult<vk::Format> {
        let formats: &[_] = if stencil {
            &[
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
                vk::Format::D16_UNORM_S8_UINT,
            ]
        } else {
            &[
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::X8_D24_UNORM_PACK32,
                vk::Format::D24_UNORM_S8_UINT,
                vk::Format::D16_UNORM,
                vk::Format::D16_UNORM_S8_UINT,
            ]
        };
        self.find_supported_format(formats, vk::ImageTiling::OPTIMAL, vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
    }

    #[inline]
    pub fn debug<F: FnOnce(&DebugUtils)>(&self, debug_f: F) {
        self.instance.debug(debug_f)
    }
}

impl std::ops::Deref for VulkanDevice {
    type Target = ash::Device;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.allocator.lock().unwrap().cleanup(AshMemoryDevice::wrap(&self.device));
            self.device.destroy_command_pool(self.transfer_pool, None);
            self.instance.surface_utils.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
        }
    }
}

#[derive(Debug)]
pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub depth_images: Vec<VkImage>,
    pub depth_imgviews: Vec<vk::ImageView>,
    pub depth_format: vk::Format,
}

impl Swapchain {
    fn new(device: &VulkanDevice, win_size: WinSize, image_count: u32, old_swapchain: vk::SwapchainKHR) -> VulkanResult<Self> {
        let surface_info = device.instance.query_surface_info(device.dev_info.phys_dev, device.surface)?;
        let surface_format = surface_info.find_surface_format();
        let present_mode = surface_info.find_present_mode(vk::PresentModeKHR::IMMEDIATE);
        let extent = surface_info.calc_extent(win_size.width, win_size.height);
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
                device.create_image_view(
                    image,
                    surface_format.format,
                    vk::ImageViewType::TYPE_2D,
                    vk::ImageAspectFlags::COLOR,
                )
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            handle: swapchain,
            images,
            image_views,
            format: surface_format.format,
            extent,
            depth_images: vec![],
            depth_imgviews: vec![],
            depth_format: vk::Format::UNDEFINED,
        })
    }

    fn create_depth_attachments(&mut self, device: &VulkanDevice, depth_format: vk::Format, image_count: usize) -> VulkanResult<()> {
        if depth_format == vk::Format::UNDEFINED {
            return Ok(());
        }
        let depth_images = (0..image_count)
            .map(|_| {
                device.allocate_image(
                    self.extent.width,
                    self.extent.height,
                    1,
                    depth_format,
                    vk::ImageCreateFlags::empty(),
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    ga::UsageFlags::FAST_DEVICE_ACCESS,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let depth_imgviews = depth_images
            .iter()
            .map(|image| device.create_image_view(**image, depth_format, vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::DEPTH))
            .collect::<Result<Vec<_>, _>>()?;

        let cmd_buffer = device.begin_one_time_commands()?;
        for img in &depth_images {
            device.transition_image_layout(
                cmd_buffer,
                img.handle,
                depth_format,
                1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            );
        }
        device.end_one_time_commands(cmd_buffer)?;

        device.debug(|d| {
            depth_images
                .iter()
                .for_each(|img| d.set_object_name(device, &img.handle, "Depth image"));
            depth_imgviews
                .iter()
                .for_each(|view| d.set_object_name(device, view, "Depth image view"));
        });

        self.depth_images = depth_images;
        self.depth_imgviews = depth_imgviews;
        self.depth_format = depth_format;
        Ok(())
    }

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
        for &imgview in &self.image_views {
            device.destroy_image_view(imgview, None);
        }
        for &img in &self.depth_imgviews {
            device.destroy_image_view(img, None);
        }
        self.depth_images.cleanup(device);
        device.swapchain_fn.destroy_swapchain(self.handle, None);
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

    #[inline]
    pub fn write_bytes(&mut self, device: &VulkanDevice, bytes: &[u8], offset: usize) -> VulkanResult<()> {
        unsafe {
            self.memory
                .write_bytes(AshMemoryDevice::wrap(&device), offset as _, bytes)
                .map_err(From::from)
        }
    }

    #[inline]
    pub fn write_slice<D>(&mut self, device: &VulkanDevice, slice: &[D], offset: usize) -> VulkanResult<()> {
        let (_, bytes, _) = unsafe { slice.align_to() };
        let byte_offset = offset * std::mem::size_of::<D>();
        self.write_bytes(device, bytes, byte_offset)
    }

    unsafe fn free_memory(&mut self, device: &VulkanDevice) {
        device
            .allocator
            .lock()
            .unwrap()
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
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_buffer(self.handle, None);
        self.free_memory(device);
    }
}

impl Cleanup<VulkanDevice> for MemoryObject<vk::Image> {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
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

    pub fn descriptor(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: *self.buffer,
            offset: 0,
            range: std::mem::size_of::<T>() as _,
        }
    }
}

impl<T> Cleanup<VulkanDevice> for UniformBuffer<T> {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        self.ub_mapped = None;
        self.buffer.memory.unmap(AshMemoryDevice::wrap(device));
        self.buffer.cleanup(device);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageData<'a> {
    Single(&'a [u8]),
    Array(u32, &'a [&'a [u8]]),
}

impl ImageData<'_> {
    fn layer_count(self) -> u32 {
        match self {
            Self::Single(_) => 1,
            Self::Array(n, _) => n,
        }
    }
}
