use crate::debug::DebugUtils;
use crate::instance::{DeviceInfo, DeviceSelection, VulkanInstance};
use crate::types::*;
use ash::extensions::khr;
use ash::vk;
use glam::UVec2;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::mem::{size_of, size_of_val};
use std::slice;
use std::sync::Mutex;

pub struct VulkanDevice {
    pub(crate) instance: VulkanInstance,
    surface: vk::SurfaceKHR,
    pub dev_info: DeviceInfo,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    transfer_pool: vk::CommandPool,
    pub swapchain_fn: khr::Swapchain,
    pub dynrender_fn: khr::DynamicRendering,
    pub pushdesc_fn: khr::PushDescriptor,
    allocator: ManuallyDrop<Mutex<Allocator>>,
}

impl VulkanDevice {
    pub(crate) fn new<W>(window: &W, app_name: &str, selection: DeviceSelection) -> VulkanResult<Self>
    where
        W: HasRawDisplayHandle + HasRawWindowHandle,
    {
        let instance = VulkanInstance::new(window, app_name)?;
        let surface = instance.create_surface(window)?;
        let dev_info = instance.pick_physical_device(surface, selection)?;
        eprintln!("Selected device: {:?}", dev_info.name);
        let device = instance.create_logical_device(&dev_info)?;
        let graphics_queue = unsafe { device.get_device_queue(dev_info.graphics_idx, 0) };
        let present_queue = unsafe { device.get_device_queue(dev_info.present_idx, 0) };
        let swapchain_fn = khr::Swapchain::new(&instance, &device);
        let dynrender_fn = khr::DynamicRendering::new(&instance, &device);
        let pushdesc_fn = khr::PushDescriptor::new(&instance, &device);

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: (*instance).clone(),
            device: device.clone(),
            physical_device: dev_info.phys_dev,
            debug_settings: Default::default(),
            buffer_device_address: false,
        })?;

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
            allocator: ManuallyDrop::new(allocator.into()),
        };

        this.transfer_pool = this.create_command_pool(family, vk::CommandPoolCreateFlags::TRANSIENT)?;

        Ok(this)
    }

    pub fn create_swapchain(&self, win_size: UVec2, image_count: u32, depth_format: vk::Format) -> VulkanResult<Swapchain> {
        let mut swapchain = Swapchain::new(self, win_size.x, win_size.y, image_count, vk::SwapchainKHR::null())?;
        swapchain.create_depth_attachments(self, depth_format, swapchain.images.len() as _)?;
        Ok(swapchain)
    }

    pub fn recreate_swapchain(&self, win_size: UVec2, old_swapchain: &Swapchain) -> VulkanResult<Swapchain> {
        let mut swapchain = Swapchain::new(
            self,
            win_size.x,
            win_size.y,
            old_swapchain.images.len() as u32,
            old_swapchain.handle,
        )?;
        swapchain.create_depth_attachments(self, old_swapchain.depth_format, swapchain.images.len() as _)?;
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

    pub fn create_pipeline_cache(&self, initial_data: &[u8]) -> VulkanResult<vk::PipelineCache> {
        let cache_ci = vk::PipelineCacheCreateInfo::builder().initial_data(initial_data);
        unsafe {
            self.device
                .create_pipeline_cache(&cache_ci, None)
                .describe_err("Failed to create pipeline cache")
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

    pub fn create_semaphore(&self, sem_type: vk::SemaphoreType) -> VulkanResult<vk::Semaphore> {
        let mut sem_type_ci = vk::SemaphoreTypeCreateInfo::builder().semaphore_type(sem_type);
        let semaphore_ci = vk::SemaphoreCreateInfo::builder().push_next(&mut sem_type_ci);
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
        &self, size: vk::DeviceSize, buf_usage: vk::BufferUsageFlags, location: MemoryLocation, name: &str,
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

        let buffer_info = vk::BufferMemoryRequirementsInfo2::builder().buffer(buffer);
        let mut ded_reqs = vk::MemoryDedicatedRequirements::default();
        let mut mem_reqs = vk::MemoryRequirements2::builder().push_next(&mut ded_reqs);
        unsafe { self.device.get_buffer_memory_requirements2(&buffer_info, &mut mem_reqs) };

        let requirements = mem_reqs.memory_requirements;
        let allocation_scheme = if ded_reqs.prefers_dedicated_allocation != vk::FALSE {
            AllocationScheme::DedicatedBuffer(buffer)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };
        let allocation = self.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme,
        })?;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .describe_err("Failed to bind buffer memory")?
        };

        Ok(VkBuffer::new(buffer, allocation))
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

    pub fn create_buffer_from_data<T: Copy>(&self, data: &[T], usage: vk::BufferUsageFlags, name: &str) -> VulkanResult<VkBuffer> {
        let size = size_of_val(data) as _;
        let mut src_buffer = self.allocate_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "Staging")?;
        let dst_buffer = self.allocate_buffer(size, vk::BufferUsageFlags::TRANSFER_DST | usage, MemoryLocation::GpuOnly, name)?;

        src_buffer.map()?.write_slice(data, 0);
        self.copy_buffer(*dst_buffer, *src_buffer, size)?;

        unsafe { src_buffer.cleanup(self) };

        Ok(dst_buffer)
    }

    pub fn allocate_image(
        &self, width: u32, height: u32, layers: u32, format: vk::Format, flags: vk::ImageCreateFlags, img_usage: vk::ImageUsageFlags,
        location: MemoryLocation, name: &str,
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

        let image_info = vk::ImageMemoryRequirementsInfo2::builder().image(image);
        let mut ded_reqs = vk::MemoryDedicatedRequirements::default();
        let mut mem_reqs = vk::MemoryRequirements2::builder().push_next(&mut ded_reqs);
        unsafe { self.device.get_image_memory_requirements2(&image_info, &mut mem_reqs) };

        let requirements = mem_reqs.memory_requirements;
        let allocation_scheme = if ded_reqs.prefers_dedicated_allocation != vk::FALSE {
            AllocationScheme::DedicatedImage(image)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };
        let allocation = self.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: false,
            allocation_scheme,
        })?;

        unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .describe_err("Failed to bind image memory")?;
        }

        Ok(VkImage::new(image, allocation))
    }

    pub fn transition_image_layout(
        &self, cmd_buffer: vk::CommandBuffer, image: vk::Image, format: vk::Format, layer_count: u32, old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        fn layout_to_access_and_stage(layout: vk::ImageLayout, is_dst: bool) -> (vk::AccessFlags, vk::PipelineStageFlags) {
            match layout {
                vk::ImageLayout::UNDEFINED if !is_dst => (vk::AccessFlags::empty(), vk::PipelineStageFlags::TOP_OF_PIPE),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::AccessFlags::TRANSFER_WRITE, vk::PipelineStageFlags::TRANSFER),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (vk::AccessFlags::SHADER_READ, vk::PipelineStageFlags::FRAGMENT_SHADER),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ),
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                ),
                vk::ImageLayout::PRESENT_SRC_KHR if is_dst => (vk::AccessFlags::empty(), vk::PipelineStageFlags::BOTTOM_OF_PIPE),
                _ => panic!("Unsupported layout transition"),
            }
        }

        let (src_access, src_stage) = layout_to_access_and_stage(old_layout, false);
        let (dst_access, dst_stage) = layout_to_access_and_stage(new_layout, true);

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
                vk::DependencyFlags::empty(),
                &[],
                &[],
                slice::from_ref(&barrier),
            );
        }
    }

    fn copy_buffer_to_image(
        &self, cmd_buffer: vk::CommandBuffer, buffer: vk::Buffer, image: vk::Image, offset: vk::Offset3D, extent: vk::Extent3D,
        layer_count: u32,
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
            .image_offset(offset)
            .image_extent(extent);
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

    pub fn create_image_from_data(
        &self, width: u32, height: u32, format: vk::Format, data: ImageData, flags: vk::ImageCreateFlags,
    ) -> VulkanResult<VkImage> {
        let layer_size = width as usize * height as usize * 4;
        let layers = data.layer_count();
        let size = layer_size as vk::DeviceSize * layers as vk::DeviceSize;
        let mut src_buffer = self.allocate_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "Staging")?;
        let tex_image = self.allocate_image(
            width,
            height,
            layers,
            format,
            flags,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            "Texture image",
        )?;

        match data {
            ImageData::Single(bytes) => {
                src_buffer.map()?.write_bytes(bytes, 0);
            }
            ImageData::Array(bytes_arr) => {
                let mut mem = src_buffer.map()?;
                for (i, &bytes) in bytes_arr.iter().enumerate() {
                    mem.write_bytes(bytes, layer_size * i);
                }
            }
        }
        let cmd_buffer = self.begin_one_time_commands()?;
        self.transition_image_layout(
            cmd_buffer,
            *tex_image,
            format,
            layers,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        self.copy_buffer_to_image(
            cmd_buffer,
            *src_buffer,
            *tex_image,
            vk::Offset3D::default(),
            vk::Extent3D { width, height, depth: 1 },
            layers,
        );
        self.transition_image_layout(
            cmd_buffer,
            *tex_image,
            format,
            layers,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        self.end_one_time_commands(cmd_buffer)?;

        unsafe { src_buffer.cleanup(self) };
        self.debug(|d| d.set_object_name(&self.device, &tex_image.handle, &format!("Texture image {width}x{height}x{layers}")));

        Ok(tex_image)
    }

    pub(crate) fn update_image_from_data(
        &self, image: vk::Image, x: i32, y: i32, width: u32, height: u32, data: ImageData,
    ) -> VulkanResult<()> {
        let layer_size = width as usize * height as usize * 4;
        let layers = data.layer_count();
        let size = layer_size as vk::DeviceSize * layers as vk::DeviceSize;
        let mut src_buffer = self.allocate_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "Staging")?;

        match data {
            ImageData::Single(bytes) => {
                src_buffer.map()?.write_bytes(bytes, 0);
            }
            ImageData::Array(bytes_arr) => {
                let mut mem = src_buffer.map()?;
                for (i, &bytes) in bytes_arr.iter().enumerate() {
                    mem.write_bytes(bytes, layer_size * i);
                }
            }
        }

        let cmd_buffer = self.begin_one_time_commands()?;
        self.transition_image_layout(
            cmd_buffer,
            image,
            vk::Format::R8G8B8A8_SRGB,
            layers,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        self.copy_buffer_to_image(
            cmd_buffer,
            *src_buffer,
            image,
            vk::Offset3D { x, y, z: 0 },
            vk::Extent3D { width, height, depth: 1 },
            layers,
        );
        self.transition_image_layout(
            cmd_buffer,
            image,
            vk::Format::R8G8B8A8_SRGB,
            layers,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        self.end_one_time_commands(cmd_buffer)?;

        unsafe { src_buffer.cleanup(self) };
        Ok(())
    }

    pub fn create_image_view(
        &self, image: vk::Image, format: vk::Format, layer: u32, view_type: vk::ImageViewType, aspect_mask: vk::ImageAspectFlags,
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
                base_array_layer: layer,
                layer_count,
            });

        unsafe {
            self.device
                .create_image_view(&imageview_ci, None)
                .describe_err("Failed to create image view")
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

    pub fn get_memory_info(&self) -> String {
        format!("{:#?}", self.allocator.lock().unwrap())
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
            ManuallyDrop::drop(&mut self.allocator);
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
    pub depth_images: Option<VkImage>,
    pub depth_imgviews: Vec<vk::ImageView>,
    pub depth_format: vk::Format,
}

impl Swapchain {
    fn new(device: &VulkanDevice, width: u32, height: u32, image_count: u32, old_swapchain: vk::SwapchainKHR) -> VulkanResult<Self> {
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
                device.create_image_view(
                    image,
                    surface_format.format,
                    0,
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
            depth_images: None,
            depth_imgviews: vec![],
            depth_format: vk::Format::UNDEFINED,
        })
    }

    fn create_depth_attachments(&mut self, device: &VulkanDevice, depth_format: vk::Format, image_count: u32) -> VulkanResult<()> {
        if depth_format == vk::Format::UNDEFINED {
            return Ok(());
        }
        let depth_images = device.allocate_image(
            self.extent.width,
            self.extent.height,
            image_count,
            depth_format,
            vk::ImageCreateFlags::empty(),
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            MemoryLocation::GpuOnly,
            "Depth image array",
        )?;

        let depth_imgviews = (0..image_count)
            .map(|layer| {
                device.create_image_view(
                    *depth_images,
                    depth_format,
                    layer,
                    vk::ImageViewType::TYPE_2D,
                    vk::ImageAspectFlags::DEPTH,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let cmd_buffer = device.begin_one_time_commands()?;
        device.transition_image_layout(
            cmd_buffer,
            *depth_images,
            depth_format,
            image_count,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );
        device.end_one_time_commands(cmd_buffer)?;

        device.debug(|d| {
            d.set_object_name(device, &*depth_images, "Depth image array");
            depth_imgviews
                .iter()
                .for_each(|view| d.set_object_name(device, view, "Depth image view"));
        });

        self.depth_images = Some(depth_images);
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

pub type VkBuffer = MemoryObject<vk::Buffer>;
pub type VkImage = MemoryObject<vk::Image>;

#[derive(Debug)]
pub struct MemoryObject<T> {
    pub handle: T,
    memory: ManuallyDrop<Allocation>,
}

impl<T> MemoryObject<T> {
    fn new(handle: T, memory: Allocation) -> Self {
        Self {
            handle,
            memory: ManuallyDrop::new(memory),
        }
    }

    #[inline]
    pub fn size(&self) -> u64 {
        self.memory.size()
    }

    pub fn map(&mut self) -> VulkanResult<MappedMemory> {
        let mapped = self.memory.mapped_slice_mut().describe_err("Failed to map memory")?;
        Ok(MappedMemory { mapped })
    }

    unsafe fn free_memory(&mut self, device: &VulkanDevice) {
        device
            .allocator
            .lock()
            .unwrap()
            .free(ManuallyDrop::take(&mut self.memory))
            .expect("Failed to free memory");
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

pub struct MappedMemory<'a> {
    mapped: &'a mut [u8],
}

impl MappedMemory<'_> {
    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8], offset: usize) {
        self.mapped[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    #[inline]
    pub fn write_slice<T: Copy>(&mut self, slice: &[T], offset: usize) {
        let (_, bytes, _) = unsafe { slice.align_to() };
        let byte_offset = offset * size_of::<T>();
        self.write_bytes(bytes, byte_offset)
    }

    #[inline]
    pub fn write_object<T: bytemuck::Pod>(&mut self, object: &T) {
        self.write_bytes(bytemuck::bytes_of(object), 0)
    }
}

pub struct UniformBuffer<T> {
    pub buffer: VkBuffer,
    _p: PhantomData<T>,
}

impl<T: bytemuck::Pod> UniformBuffer<T> {
    pub fn new(device: &VulkanDevice) -> VulkanResult<Self> {
        let size = size_of::<T>() as _;
        let buffer = device.allocate_buffer(
            size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "Uniform buffer",
        )?;
        device.debug(|d| d.set_object_name(device, &*buffer, "Uniform buffer"));
        Ok(Self { buffer, _p: PhantomData })
    }

    pub fn write_uniforms(&mut self, ubo: T) -> VulkanResult<()> {
        self.buffer.map()?.write_object(&ubo);
        Ok(())
    }

    pub fn descriptor(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: *self.buffer,
            offset: 0,
            range: size_of::<T>() as _,
        }
    }
}

impl<T> Cleanup<VulkanDevice> for UniformBuffer<T> {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        self.buffer.cleanup(device);
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ImageData<'a> {
    Single(&'a [u8]),
    Array(&'a [&'a [u8]]),
}

impl ImageData<'_> {
    fn layer_count(self) -> u32 {
        match self {
            Self::Single(_) => 1,
            Self::Array(s) => s.len() as _,
        }
    }
}
