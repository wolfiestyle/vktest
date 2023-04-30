use crate::create::CreateFromInfo;
use crate::debug::DebugUtils;
use crate::format::{image_aspect_flags, FormatInfo};
use crate::instance::{DeviceInfo, DeviceSelection, VulkanInstance};
use crate::types::*;
use ash::extensions::khr;
use ash::vk;
use glam::UVec2;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::mem::ManuallyDrop;
use std::mem::{size_of, size_of_val};
use std::slice;
use std::sync::Mutex;

pub struct VulkanDevice {
    pub(crate) instance: VulkanInstance,
    pub(crate) surface: vk::SurfaceKHR,
    pub dev_info: DeviceInfo,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    transfer_pool: vk::CommandPool,
    transfer_fence: vk::Fence,
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
        eprintln!("Selected device: {}\nDriver: {}", dev_info.name, dev_info.driver);
        let device = instance.create_logical_device(&dev_info)?;
        let graphics_queue = unsafe { device.get_device_queue(dev_info.graphics_idx, 0) };
        let present_queue = unsafe { device.get_device_queue(dev_info.present_idx, 0) };
        let swapchain_fn = khr::Swapchain::new(&instance, &device);
        let dynrender_fn = khr::DynamicRendering::new(&instance, &device);
        let pushdesc_fn = khr::PushDescriptor::new(&instance, &device);

        let transfer_pool = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(dev_info.graphics_idx)
            .create(&device)?;
        let transfer_fence = vk::FenceCreateInfo::builder().create(&device)?;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: (*instance).clone(),
            device: device.clone(),
            physical_device: dev_info.phys_dev,
            debug_settings: Default::default(),
            buffer_device_address: false,
        })?;

        let this = Self {
            instance,
            surface,
            dev_info,
            device,
            graphics_queue,
            present_queue,
            transfer_pool,
            transfer_fence,
            swapchain_fn,
            dynrender_fn,
            pushdesc_fn,
            allocator: ManuallyDrop::new(allocator.into()),
        };

        Ok(this)
    }

    pub fn make_semaphore(&self, sem_type: vk::SemaphoreType) -> VulkanResult<vk::Semaphore> {
        let mut sem_type_ci = vk::SemaphoreTypeCreateInfo::builder().semaphore_type(sem_type);
        vk::SemaphoreCreateInfo::builder().push_next(&mut sem_type_ci).create(&self.device)
    }

    pub(crate) fn begin_one_time_commands(&self) -> VulkanResult<vk::CommandBuffer> {
        let cmd_buffer = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.transfer_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)
            .create(&self.device)?[0];
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .describe_err("Failed to begin recording command buffer")?;
        }
        Ok(cmd_buffer)
    }

    pub(crate) fn end_one_time_commands(&self, cmd_buffer: vk::CommandBuffer) -> VulkanResult<()> {
        let submit_info = vk::SubmitInfo::builder().command_buffers(slice::from_ref(&cmd_buffer));
        unsafe {
            self.device
                .end_command_buffer(cmd_buffer)
                .describe_err("Failed to end recording command buffer")?;
            self.device
                .queue_submit(self.graphics_queue, slice::from_ref(&submit_info), self.transfer_fence)
                .describe_err("Failed to submit queue")?;
            self.device
                .wait_for_fences(slice::from_ref(&self.transfer_fence), true, u64::MAX)
                .describe_err("Failed to wait fence")?;
            self.device
                .reset_fences(slice::from_ref(&self.transfer_fence))
                .describe_err("Failed to reset fence")?;
            self.device.free_command_buffers(self.transfer_pool, slice::from_ref(&cmd_buffer));
        }
        Ok(())
    }

    fn allocate_buffer(
        &self, size: vk::DeviceSize, buf_usage: vk::BufferUsageFlags, location: MemoryLocation, name: &str,
    ) -> VulkanResult<MemoryObject<vk::Buffer, MemoryLocation>> {
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

        Ok(MemoryObject {
            handle: buffer,
            memory: ManuallyDrop::new(allocation),
            size,
            props: location,
        })
    }

    #[inline]
    pub fn allocate_cpu_buffer(&self, size: vk::DeviceSize, buf_usage: vk::BufferUsageFlags, name: &str) -> VulkanResult<VkBuffer> {
        self.allocate_buffer(size, buf_usage, MemoryLocation::CpuToGpu, name)
    }

    #[inline]
    pub fn allocate_gpu_buffer(&self, size: vk::DeviceSize, buf_usage: vk::BufferUsageFlags, name: &str) -> VulkanResult<VkBuffer> {
        self.allocate_buffer(size, buf_usage, MemoryLocation::GpuOnly, name)
    }

    pub fn copy_buffer(&self, src_buffer: &VkBuffer, dst_buffer: &VkBuffer, dst_offset: u64) -> VulkanResult<()> {
        if src_buffer.size() > dst_offset + dst_buffer.size() {
            return VkError::InvalidArgument("Buffer write out of bounds").into();
        }
        let cmd_buffer = self.begin_one_time_commands()?;
        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset,
            size: src_buffer.size(),
        };
        unsafe {
            self.device
                .cmd_copy_buffer(cmd_buffer, src_buffer.handle, dst_buffer.handle, slice::from_ref(&copy_region));
        }
        self.end_one_time_commands(cmd_buffer)
    }

    pub fn create_buffer_from_data<T: Copy>(&self, data: &[T], usage: vk::BufferUsageFlags, name: &str) -> VulkanResult<VkBuffer> {
        let size = size_of_val(data) as _;
        let mut src_buffer = self.allocate_cpu_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC, "Staging")?;
        let dst_buffer = self.allocate_gpu_buffer(size, vk::BufferUsageFlags::TRANSFER_DST | usage, name)?;

        src_buffer.map()?.write_slice(data, 0);
        self.copy_buffer(&src_buffer, &dst_buffer, 0)?;

        self.dispose_of(src_buffer);
        Ok(dst_buffer)
    }

    pub fn allocate_image(
        &self, params: ImageParams, flags: vk::ImageCreateFlags, img_usage: vk::ImageUsageFlags, location: MemoryLocation, name: &str,
    ) -> VulkanResult<VkImage> {
        let image_ci = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: params.width,
                height: params.height,
                depth: 1,
            })
            .array_layers(params.layers)
            .mip_levels(params.mip_levels)
            .samples(params.samples)
            .format(params.format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(img_usage)
            .flags(flags)
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

        Ok(VkImage {
            handle: image,
            memory: ManuallyDrop::new(allocation),
            size: requirements.size,
            props: params,
        })
    }

    fn layout_to_access_and_stage(layout: vk::ImageLayout, is_dst: bool) -> (vk::AccessFlags, vk::PipelineStageFlags) {
        match layout {
            vk::ImageLayout::UNDEFINED if !is_dst => (vk::AccessFlags::empty(), vk::PipelineStageFlags::TOP_OF_PIPE),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (vk::AccessFlags::TRANSFER_READ, vk::PipelineStageFlags::TRANSFER),
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
            vk::ImageLayout::GENERAL => (vk::AccessFlags::SHADER_WRITE, vk::PipelineStageFlags::COMPUTE_SHADER),
            vk::ImageLayout::PRESENT_SRC_KHR if is_dst => (vk::AccessFlags::empty(), vk::PipelineStageFlags::BOTTOM_OF_PIPE),
            _ => panic!("Unsupported layout transition"),
        }
    }

    pub fn transition_image_layout(
        &self, cmd_buffer: vk::CommandBuffer, image: vk::Image, subresource: vk::ImageSubresourceRange, old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let (src_access, src_stage) = Self::layout_to_access_and_stage(old_layout, false);
        let (dst_access, dst_stage) = Self::layout_to_access_and_stage(new_layout, true);

        let barrier = vk::ImageMemoryBarrier::builder()
            .image(image)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(subresource);
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

    pub(crate) fn image_reuse_barrier(&self, cmd_buffer: vk::CommandBuffer, image: vk::Image, format: vk::Format, layout: vk::ImageLayout) {
        let (src_access, src_stage) = match layout {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            ),
            _ => panic!("Unsupported image barrier"),
        };
        let (dst_access, dst_stage) = Self::layout_to_access_and_stage(layout, true);

        let barrier = vk::ImageMemoryBarrier::builder()
            .image(image)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(layout)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: image_aspect_flags(format),
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
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

    pub(crate) unsafe fn generate_mipmaps(&self, cmd_buffer: vk::CommandBuffer, image: vk::Image, params: ImageParams) {
        let aspect_mask = params.aspect_flags();
        let mut barrier = vk::ImageMemoryBarrier::builder()
            .image(image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: params.layers,
            });
        let mut width = params.width;
        let mut height = params.height;
        for i in 1..params.mip_levels {
            // prepare source mip level for reading
            barrier.subresource_range.base_mip_level = i - 1;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;
            self.device.cmd_pipeline_barrier(
                cmd_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                slice::from_ref(&barrier),
            );
            // perform the blit on the destination mip level
            let width_2 = 1.max(width / 2);
            let height_2 = 1.max(height / 2);
            let blit = vk::ImageBlit::builder()
                .src_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: width as _,
                        y: height as _,
                        z: 1,
                    },
                ])
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: params.layers,
                })
                .dst_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: width_2 as _,
                        y: height_2 as _,
                        z: 1,
                    },
                ])
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: params.layers,
                });
            self.device.cmd_blit_image(
                cmd_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                slice::from_ref(&blit),
                vk::Filter::LINEAR,
            );
            // transition the source level into shader read
            barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
            self.device.cmd_pipeline_barrier(
                cmd_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                slice::from_ref(&barrier),
            );
            // use the current destination image as the source for the next iteration
            width = width_2;
            height = height_2;
        }
        // transition the last mip level
        barrier.subresource_range.base_mip_level = params.mip_levels - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
        self.device.cmd_pipeline_barrier(
            cmd_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            slice::from_ref(&barrier),
        );
    }

    pub fn create_image_from_data(&self, params: ImageParams, data: ImageData, flags: vk::ImageCreateFlags) -> VulkanResult<VkImage> {
        if params.layers != data.layer_count() {
            return VkError::InvalidArgument("Image and data layer count doesn't match").into();
        }
        if params.samples != vk::SampleCountFlags::TYPE_1 {
            return VkError::InvalidArgument("Creating multisampled texture is not supported").into();
        }
        // check if this image format supports linear filtering (for mipmap generation)
        if params.mip_levels > 1 {
            let linear_filter_supp = unsafe {
                self.instance
                    .get_physical_device_format_properties(self.dev_info.phys_dev, params.format)
                    .optimal_tiling_features
                    .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
            };
            if !linear_filter_supp {
                //FIXME: fallback to another method
                return VkError::EngineError("Image linear filter not supported").into();
            }
        }
        let fmt_info = FormatInfo::try_from(params.format)?;
        let layer_size = params.width as usize * params.height as usize * fmt_info.size;
        let size = layer_size as vk::DeviceSize * params.layers as vk::DeviceSize;
        if !data.check_size(layer_size) {
            return VkError::InvalidArgument("Image data size doesn't match the expected size").into();
        }
        // create staging buffer that will contain the image bytes
        let mut src_buffer = self.allocate_cpu_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC, "Staging")?;
        // create destination image that will be usable from shaders
        let tex_image = self.allocate_image(
            params,
            flags,
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            "Texture image",
        )?;
        // write image data to the staging buffer
        data.write_to_buffer(&mut src_buffer.map()?);
        // setup all layers and mip levels for transfer
        let cmd_buffer = self.begin_one_time_commands()?;
        self.transition_image_layout(
            cmd_buffer,
            *tex_image,
            params.subresource_range(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        unsafe {
            // copy the image bytes into the base mip level
            let region = vk::BufferImageCopy::builder()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: params.aspect_flags(),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: params.layers,
                })
                .image_offset(vk::Offset3D::default())
                .image_extent(params.extent());
            self.device.cmd_copy_buffer_to_image(
                cmd_buffer,
                *src_buffer,
                *tex_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                slice::from_ref(&region),
            );
            // generate mipmaps and set image to the final layout
            self.generate_mipmaps(cmd_buffer, *tex_image, params);
        }
        self.end_one_time_commands(cmd_buffer)?;

        self.dispose_of(src_buffer);
        self.debug(|d| d.set_object_name(&self.device, &tex_image.handle, &format!("Texture image {params:?}")));

        Ok(tex_image)
    }

    pub fn update_image_from_data(&self, image: &VkImage, pos: UVec2, dims: UVec2, base_layer: u32, data: ImageData) -> VulkanResult<()> {
        let layers = data.layer_count();
        if base_layer + layers > image.props.layers {
            return VkError::InvalidArgument("Data layer count out of bounds").into();
        }
        let my_dims = UVec2::new(image.props.width, image.props.height);
        if pos.cmpge(my_dims).any() || (pos + dims).cmpge(my_dims).any() {
            return VkError::InvalidArgument("Image update rect out of bounds").into();
        }
        let fmt_info = FormatInfo::try_from(image.props.format)?;
        let layer_size = dims.x as usize * dims.y as usize * fmt_info.size;
        let size = layer_size as vk::DeviceSize * layers as vk::DeviceSize;
        if !data.check_size(layer_size) {
            return VkError::InvalidArgument("Image data size doesn't match the expected size").into();
        }

        let mut src_buffer = self.allocate_cpu_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC, "Staging")?;
        data.write_to_buffer(&mut src_buffer.map()?);

        let cmd_buffer = self.begin_one_time_commands()?;
        // transition everything because we're gonna recreate mipmaps after
        self.transition_image_layout(
            cmd_buffer,
            image.handle,
            image.props.subresource_range(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        unsafe {
            // update base mip
            let region = vk::BufferImageCopy::builder()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: image.props.aspect_flags(),
                    mip_level: 0,
                    base_array_layer: base_layer,
                    layer_count: layers,
                })
                .image_offset(vk::Offset3D {
                    x: pos.x as _,
                    y: pos.y as _,
                    z: 0,
                })
                .image_extent(vk::Extent3D {
                    width: dims.x,
                    height: dims.y,
                    depth: 1,
                });
            self.device.cmd_copy_buffer_to_image(
                cmd_buffer,
                *src_buffer,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                slice::from_ref(&region),
            );
            // re-generate mips and put image back in shader read layout
            self.generate_mipmaps(cmd_buffer, image.handle, image.props);
        }
        self.end_one_time_commands(cmd_buffer)?;

        self.dispose_of(src_buffer);
        Ok(())
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

    #[inline]
    pub fn dispose_of(&self, mut obj: impl Cleanup<Self>) {
        unsafe { obj.cleanup(self) };
        std::mem::forget(obj);
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
            self.device.destroy_fence(self.transfer_fence, None);
            self.instance.surface_utils.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
        }
    }
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("instance", &self.instance)
            .field("surface", &self.surface)
            .field("dev_info", &self.dev_info)
            .field("graphics_queue", &self.graphics_queue)
            .field("present_queue", &self.present_queue)
            .field("transfer_pool", &self.transfer_pool)
            .field("transfer_fence", &self.transfer_fence)
            .finish_non_exhaustive()
    }
}

impl Cleanup<VulkanDevice> for vk::CommandPool {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_command_pool(*self, None);
    }
}

impl Cleanup<VulkanDevice> for vk::Sampler {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_sampler(*self, None);
    }
}

impl Cleanup<VulkanDevice> for vk::DescriptorSetLayout {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_descriptor_set_layout(*self, None);
    }
}

impl Cleanup<VulkanDevice> for vk::ImageView {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_image_view(*self, None);
    }
}

pub type VkBuffer = MemoryObject<vk::Buffer, MemoryLocation>;
pub type VkImage = MemoryObject<vk::Image, ImageParams>;

#[derive(Debug)]
pub struct MemoryObject<T, P> {
    pub handle: T,
    memory: ManuallyDrop<Allocation>,
    size: u64,
    pub props: P,
}

impl<T, P> MemoryObject<T, P> {
    #[inline]
    pub fn size(&self) -> u64 {
        self.size
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

impl<T, P> std::ops::Deref for MemoryObject<T, P> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl<P> Cleanup<VulkanDevice> for MemoryObject<vk::Buffer, P> {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_buffer(self.handle, None);
        self.free_memory(device);
    }
}

impl<P> Cleanup<VulkanDevice> for MemoryObject<vk::Image, P> {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_image(self.handle, None);
        self.free_memory(device);
    }
}

impl<P> MemoryObject<vk::Buffer, P> {
    pub fn descriptor(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.handle,
            offset: 0,
            range: self.size(),
        }
    }

    pub fn descriptor_slice(&self, offset: u64, range: u64) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.handle,
            offset,
            range,
        }
    }
}

impl MemoryObject<vk::Image, ImageParams> {
    #[inline]
    pub fn create_view(&self, device: &ash::Device, view_type: vk::ImageViewType) -> VulkanResult<vk::ImageView> {
        self.create_view_subresource(device, view_type, self.props.subresource_range())
    }

    pub fn create_view_subresource(
        &self, device: &ash::Device, view_type: vk::ImageViewType, subresource: vk::ImageSubresourceRange,
    ) -> VulkanResult<vk::ImageView> {
        vk::ImageViewCreateInfo::builder()
            .image(self.handle)
            .format(self.props.format)
            .view_type(view_type)
            .subresource_range(subresource)
            .create(device)
    }

    pub fn create_view_swizzle(
        &self, device: &ash::Device, view_type: vk::ImageViewType, swizzle: vk::ComponentMapping,
    ) -> VulkanResult<vk::ImageView> {
        vk::ImageViewCreateInfo::builder()
            .image(self.handle)
            .format(self.props.format)
            .view_type(view_type)
            .subresource_range(self.props.subresource_range())
            .components(swizzle)
            .create(device)
    }
}

#[derive(Debug)]
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
    pub fn write_object<T: bytemuck::Pod>(&mut self, object: &T, offset: usize) {
        let byte_offset = offset * size_of::<T>();
        self.write_bytes(bytemuck::bytes_of(object), byte_offset)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CubeData<'a> {
    posx: &'a [u8],
    negx: &'a [u8],
    posy: &'a [u8],
    negy: &'a [u8],
    posz: &'a [u8],
    negz: &'a [u8],
}

impl<'a> CubeData<'a> {
    pub fn from_array(array: [&'a [u8]; 6]) -> Self {
        CubeData {
            posx: array[0],
            negx: array[1],
            posy: array[2],
            negy: array[3],
            posz: array[4],
            negz: array[5],
        }
    }

    pub fn try_from_iter(iter: impl IntoIterator<Item = &'a [u8]>) -> Option<Self> {
        let mut iter = iter.into_iter();
        Some(CubeData {
            posx: iter.next()?,
            negx: iter.next()?,
            posy: iter.next()?,
            negy: iter.next()?,
            posz: iter.next()?,
            negz: iter.next()?,
        })
    }

    fn check_size(&self, layer_size: usize) -> bool {
        self.posx.len() == layer_size
            && self.negx.len() == layer_size
            && self.posy.len() == layer_size
            && self.negy.len() == layer_size
            && self.posz.len() == layer_size
            && self.negz.len() == layer_size
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ImageData<'a> {
    Single(&'a [u8]),
    Array(&'a [&'a [u8]]),
    Cube(CubeData<'a>),
}

impl<'a> ImageData<'a> {
    pub fn layer_count(&self) -> u32 {
        match self {
            Self::Single(_) => 1,
            Self::Array(s) => s.len() as _,
            Self::Cube(_) => 6,
        }
    }

    fn check_size(&self, layer_size: usize) -> bool {
        match self {
            Self::Single(data) => data.len() == layer_size,
            Self::Array(data_arr) => data_arr.iter().all(|data| data.len() == layer_size),
            Self::Cube(cube) => cube.check_size(layer_size),
        }
    }

    pub(crate) fn image_create_flags(&self) -> vk::ImageCreateFlags {
        match self {
            Self::Cube(_) => vk::ImageCreateFlags::CUBE_COMPATIBLE,
            _ => vk::ImageCreateFlags::empty(),
        }
    }

    pub(crate) fn view_type(&self) -> vk::ImageViewType {
        match self {
            Self::Single(_) | Self::Array(_) => vk::ImageViewType::TYPE_2D,
            Self::Cube(_) => vk::ImageViewType::CUBE,
        }
    }

    #[rustfmt::skip]
    fn write_to_buffer(self, mem: &mut MappedMemory) {
        match self {
            ImageData::Single(bytes) => {
                mem.write_bytes(bytes, 0);
            }
            ImageData::Array(bytes_arr) => {
                let mut offset = 0;
                for &bytes in bytes_arr.iter() {
                    mem.write_bytes(bytes, offset);
                    offset += bytes.len();
                }
            }
            ImageData::Cube(CubeData { posx, negx, posy, negy, posz, negz })=> {
                let mut offset = 0;
                for bytes in [posx, negx, posy, negy, posz, negz] {
                    mem.write_bytes(bytes, offset);
                    offset += bytes.len();
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageParams {
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub mip_levels: u32,
    pub samples: vk::SampleCountFlags,
    pub format: vk::Format,
}

impl Default for ImageParams {
    #[inline]
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            layers: 1,
            mip_levels: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            format: vk::Format::UNDEFINED,
        }
    }
}

impl ImageParams {
    #[inline]
    pub fn size(&self) -> UVec2 {
        UVec2 {
            x: self.width,
            y: self.height,
        }
    }

    #[inline]
    pub fn extent(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.width,
            height: self.height,
            depth: 1,
        }
    }

    #[inline]
    pub fn aspect_flags(&self) -> vk::ImageAspectFlags {
        image_aspect_flags(self.format)
    }

    #[inline]
    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect_flags(),
            base_mip_level: 0,
            level_count: self.mip_levels,
            base_array_layer: 0,
            layer_count: self.layers,
        }
    }
}
