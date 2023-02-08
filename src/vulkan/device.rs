use crate::vulkan::engine::Vertex;
use crate::vulkan::instance::{DeviceInfo, SurfaceInfo, VulkanInstance};
use crate::vulkan::types::*;
use ash::extensions::khr;
use ash::vk;
use cstr::cstr;
use winit::window::Window;

pub struct VulkanDevice {
    instance: VulkanInstance,
    surface: vk::SurfaceKHR,
    pub dev_info: DeviceInfo,
    pub surface_info: SurfaceInfo,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub swapchain_utils: khr::Swapchain,
}

impl VulkanDevice {
    pub fn new(window: &Window) -> VulkanResult<Self> {
        let vk = VulkanInstance::new(window)?;
        let surface = vk.create_surface(window)?;
        let dev_info = vk.pick_physical_device(surface)?;
        eprintln!("Selected device: {:?}", dev_info.name);
        let surface_info = vk.query_surface_info(dev_info.phys_dev, surface)?;
        let device = vk.create_logical_device(&dev_info)?;
        let graphics_queue = unsafe { device.get_device_queue(dev_info.graphics_idx, 0) };
        let present_queue = unsafe { device.get_device_queue(dev_info.present_idx, 0) };
        let swapchain_utils = khr::Swapchain::new(&vk.instance, &device);

        Ok(Self {
            instance: vk,
            surface,
            dev_info,
            surface_info,
            device,
            graphics_queue,
            present_queue,
            swapchain_utils,
        })
    }

    pub fn create_swapchain(
        &self, window: &Window, image_count: u32, old_swapchain: Option<vk::SwapchainKHR>,
    ) -> VulkanResult<SwapchainInfo> {
        let surface_format = self.surface_info.surface_format();
        let present_mode = self.surface_info.present_mode();
        let win_size = window.inner_size();
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
            .map(|&image| {
                let imageview_ci = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);

                unsafe { self.device.create_image_view(&imageview_ci, None) }
            })
            .collect::<Result<_, _>>()
            .describe_err("Failed to create image views")?;

        Ok(SwapchainInfo {
            handle: swapchain,
            //images,
            image_views,
            format: surface_format.format,
            extent,
        })
    }

    fn create_shader_module(&self, spirv_code: &[u32]) -> VulkanResult<vk::ShaderModule> {
        let shader_ci = vk::ShaderModuleCreateInfo::builder().code(spirv_code);

        unsafe {
            self.device
                .create_shader_module(&shader_ci, None)
                .describe_err("Failed to create shader module")
        }
    }

    pub fn create_render_pass(&self, format: vk::Format) -> VulkanResult<vk::RenderPass> {
        let color_attach = [vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        }];

        let attach_ref = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let subpass = [vk::SubpassDescription::builder().color_attachments(&attach_ref).build()];

        let dependency = [vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .build()];

        let render_pass_ci = vk::RenderPassCreateInfo::builder()
            .attachments(&color_attach)
            .subpasses(&subpass)
            .dependencies(&dependency);

        unsafe {
            self.device
                .create_render_pass(&render_pass_ci, None)
                .describe_err("Failed to create render pass")
        }
    }

    pub fn create_graphics_pipeline(
        &self, vert_shader_spv: &[u32], frag_shader_spv: &[u32], render_pass: vk::RenderPass,
    ) -> VulkanResult<(vk::Pipeline, vk::PipelineLayout)> {
        let vert_shader = self.create_shader_module(vert_shader_spv)?;
        let frag_shader = self.create_shader_module(frag_shader_spv)?;

        let entry_point = cstr!("main");
        let shader_stages_ci = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader)
                .name(entry_point)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader)
                .name(entry_point)
                .build(),
        ];

        let binding_desc = [Vertex::binding_desc(0)];
        let attr_desc = Vertex::attr_desc(0);

        let vertex_input_ci = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_desc)
            .vertex_attribute_descriptions(&attr_desc);

        let input_assembly_ci = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state_ci = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let dynamic_state_ci =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let rasterizer_ci = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE);

        let multisample_ci = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0);

        let color_attach = [vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .build()];

        let color_blend_ci = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_attach);

        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_ci, None)
                .describe_err("Failed to create pipeline layout")?
        };

        let pipeline_ci = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages_ci)
            .vertex_input_state(&vertex_input_ci)
            .input_assembly_state(&input_assembly_ci)
            .viewport_state(&viewport_state_ci)
            .rasterization_state(&rasterizer_ci)
            .multisample_state(&multisample_ci)
            .color_blend_state(&color_blend_ci)
            .dynamic_state(&dynamic_state_ci)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .build();

        let pipeline = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
                .map_err(|(_, err)| VkError::VulkanMsg("Error creating pipeline", err))?
        };

        unsafe {
            self.device.destroy_shader_module(vert_shader, None);
            self.device.destroy_shader_module(frag_shader, None);
        }

        Ok((pipeline[0], pipeline_layout))
    }

    pub fn create_framebuffers(&self, swapchain: &SwapchainInfo, render_pass: vk::RenderPass) -> VulkanResult<Vec<vk::Framebuffer>> {
        swapchain
            .image_views
            .iter()
            .map(|&imgview| {
                let attachments = [imgview];
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

    pub fn create_command_pool(&self, flags: vk::CommandPoolCreateFlags) -> VulkanResult<vk::CommandPool> {
        let command_pool_ci = vk::CommandPoolCreateInfo::builder()
            .flags(flags)
            .queue_family_index(self.dev_info.graphics_idx);

        unsafe {
            self.device
                .create_command_pool(&command_pool_ci, None)
                .describe_err("Failed to create command pool")
        }
    }

    pub fn create_command_buffers(&self, pool: vk::CommandPool, count: u32) -> VulkanResult<Vec<vk::CommandBuffer>> {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .describe_err("Failed to allocate command buffers")
        }
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

    pub fn wait_idle(&self) -> VulkanResult<()> {
        unsafe { self.device.device_wait_idle().describe_err("Failed to wait device idle") }
    }

    pub fn update_surface_info(&mut self) -> VulkanResult<()> {
        self.surface_info = self.instance.query_surface_info(self.dev_info.phys_dev, self.surface)?;
        Ok(())
    }

    fn allocate_buffer(
        &self, size: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags,
    ) -> VulkanResult<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_ci = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_ci, None)
                .describe_err("Failed to create buffer")?
        };

        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mem_type = self.dev_info.find_memory_type(mem_reqs.memory_type_bits, properties)?;

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type);

        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .describe_err("Failed to allocate buffer memory")?
        };
        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .describe_err("Failed to bind buffer memory")?
        };

        Ok((buffer, memory))
    }

    fn write_memory<T: Copy>(&self, memory: vk::DeviceMemory, data: &[T]) -> VulkanResult<()> {
        let size = std::mem::size_of_val(data) as _;
        unsafe {
            let mapped_ptr = self
                .device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                .describe_err("Failed to map buffer memory")? as *mut T;
            std::slice::from_raw_parts_mut(mapped_ptr, data.len()).copy_from_slice(data);
            self.device.unmap_memory(memory);
        };
        Ok(())
    }

    fn copy_buffer(&self, dst_buffer: vk::Buffer, src_buffer: vk::Buffer, size: vk::DeviceSize, pool: vk::CommandPool) -> VulkanResult<()> {
        let cmd_buffer = self.create_command_buffers(pool, 1)?;
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer[0], &begin_info)
                .describe_err("Failed to begin recording command buffer")?;
            self.device.cmd_copy_buffer(cmd_buffer[0], src_buffer, dst_buffer, &[copy_region]);
            self.device
                .end_command_buffer(cmd_buffer[0])
                .describe_err("Failed to end recording command buffer")?;
        }

        let submit_info = [vk::SubmitInfo::builder().command_buffers(&cmd_buffer).build()];
        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &submit_info, vk::Fence::null())
                .describe_err("Failed to submit queue")?;
            self.device
                .queue_wait_idle(self.graphics_queue)
                .describe_err("Failed to wait queue idle")?;
            self.device.free_command_buffers(pool, &cmd_buffer);
        }

        Ok(())
    }

    pub fn create_buffer<T: Copy>(
        &self, data: &[T], usage: vk::BufferUsageFlags, pool: vk::CommandPool,
    ) -> VulkanResult<(vk::Buffer, vk::DeviceMemory)> {
        let size = std::mem::size_of_val(data) as _;
        let (src_buffer, src_memory) = self.allocate_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let (dst_buffer, dst_memory) = self.allocate_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        self.write_memory(src_memory, data)?;
        self.copy_buffer(dst_buffer, src_buffer, size, pool)?;

        unsafe {
            self.device.destroy_buffer(src_buffer, None);
            self.device.free_memory(src_memory, None);
        }

        Ok((dst_buffer, dst_memory))
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

    pub unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        for &imgview in &self.image_views {
            device.destroy_image_view(imgview, None);
        }
        device.swapchain_utils.destroy_swapchain(self.handle, None);
    }
}
