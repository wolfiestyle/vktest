use crate::device::{SwapchainInfo, VulkanDevice};
use crate::types::*;
use ash::vk;
use cgmath::{Deg, Matrix4, Point3, Vector3};
use cstr::cstr;
use inline_spirv::include_spirv;
use std::array;
use std::time::Instant;
use winit::window::Window;

const SWAPCHAIN_IMAGE_COUNT: u32 = 3;
const MAX_FRAMES_IN_FLIGHT: usize = 2;
type Vertex = ([f32; 2], [f32; 3]);

pub struct VulkanApp {
    device: VulkanDevice,
    swapchain: SwapchainInfo,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    sync: Vec<FrameSyncState>,
    current_frame: usize,
    vertex_buffer: vk::Buffer,
    vb_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    ib_memory: vk::DeviceMemory,
    uniforms: Vec<UniformData>,
    start_time: Instant,
}

impl VulkanApp {
    pub fn new(window: &Window) -> VulkanResult<Self> {
        let vk = VulkanDevice::new(window)?;
        let swapchain = vk.create_swapchain(window, SWAPCHAIN_IMAGE_COUNT, None)?;

        let vert_spv = include_spirv!("src/shaders/color.vert.glsl", vert, glsl);
        let frag_spv = include_spirv!("src/shaders/color.frag.glsl", frag, glsl);
        let render_pass = Self::create_render_pass(&vk, swapchain.format)?;
        let framebuffers = vk.create_framebuffers(&swapchain, render_pass)?;
        let descriptor_layout = Self::create_descriptor_set_layout(&vk)?;
        let descriptor_pool = vk.create_descriptor_pool(MAX_FRAMES_IN_FLIGHT as u32)?;
        let descriptor_sets = vk.create_descriptor_sets(descriptor_pool, &[descriptor_layout; MAX_FRAMES_IN_FLIGHT])?;
        let (pipeline, pipeline_layout) = Self::create_graphics_pipeline(&vk, vert_spv, frag_spv, render_pass, descriptor_layout)?;

        let command_pool = vk.create_command_pool(vk.dev_info.graphics_idx, vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)?;
        let command_buffers = vk.create_command_buffers(command_pool, MAX_FRAMES_IN_FLIGHT as u32)?;
        let sync = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| FrameSyncState::new(&vk))
            .collect::<Result<_, _>>()?;

        let vertices: [Vertex; 4] = [
            ([-0.5, -0.5], [1.0, 0.0, 0.0]),
            ([0.5, -0.5], [0.0, 1.0, 0.0]),
            ([0.5, 0.5], [0.0, 0.0, 1.0]),
            ([-0.5, 0.5], [1.0, 1.0, 1.0]),
        ];
        let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];
        let (vertex_buffer, vb_memory) = vk.create_buffer(&vertices, vk::BufferUsageFlags::VERTEX_BUFFER, command_pool)?;
        let (index_buffer, ib_memory) = vk.create_buffer(&indices, vk::BufferUsageFlags::INDEX_BUFFER, command_pool)?;
        let uniforms = (0..MAX_FRAMES_IN_FLIGHT).map(|_| UniformData::new(&vk)).collect::<Result<_, _>>()?;

        let this = Self {
            device: vk,
            swapchain,
            render_pass,
            pipeline,
            pipeline_layout,
            descriptor_layout,
            descriptor_pool,
            descriptor_sets,
            framebuffers,
            command_pool,
            command_buffers,
            sync,
            current_frame: 0,
            vertex_buffer,
            vb_memory,
            index_buffer,
            ib_memory,
            uniforms,
            start_time: Instant::now(),
        };

        this.populate_descriptor_sets();

        Ok(this)
    }

    fn create_render_pass(device: &VulkanDevice, format: vk::Format) -> VulkanResult<vk::RenderPass> {
        let color_attach = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        };

        let attach_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription::builder()
            .color_attachments(array::from_ref(&attach_ref))
            .build();

        let dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .build();

        let render_pass_ci = vk::RenderPassCreateInfo::builder()
            .attachments(array::from_ref(&color_attach))
            .subpasses(array::from_ref(&subpass))
            .dependencies(array::from_ref(&dependency));

        unsafe {
            device
                .create_render_pass(&render_pass_ci, None)
                .describe_err("Failed to create render pass")
        }
    }

    fn create_descriptor_set_layout(device: &VulkanDevice) -> VulkanResult<vk::DescriptorSetLayout> {
        let layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build();

        let desc_layout_ci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(array::from_ref(&layout_binding));

        unsafe {
            device
                .create_descriptor_set_layout(&desc_layout_ci, None)
                .describe_err("Failed to create descriptor set layout")
        }
    }

    fn populate_descriptor_sets(&self) {
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(self.uniforms[i].uniform_buffer)
                .offset(0)
                .range(std::mem::size_of::<UniformBufferObject>() as _)
                .build();
            let descr_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(array::from_ref(&buffer_info))
                .build();
            unsafe {
                self.device.update_descriptor_sets(array::from_ref(&descr_write), &[]);
            }
        }
    }

    fn create_graphics_pipeline(
        device: &VulkanDevice, vert_shader_spv: &[u32], frag_shader_spv: &[u32], render_pass: vk::RenderPass,
        descriptor_layout: vk::DescriptorSetLayout,
    ) -> VulkanResult<(vk::Pipeline, vk::PipelineLayout)> {
        let vert_shader = device.create_shader_module(vert_shader_spv)?;
        let frag_shader = device.create_shader_module(frag_shader_spv)?;

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

        let color_attach = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .build();

        let color_blend_ci = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(array::from_ref(&color_attach));

        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::builder().set_layouts(array::from_ref(&descriptor_layout));

        let pipeline_layout = unsafe {
            device
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
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
                .map_err(|(_, err)| VkError::VulkanMsg("Error creating pipeline", err))?
        };

        unsafe {
            device.destroy_shader_module(vert_shader, None);
            device.destroy_shader_module(frag_shader, None);
        }

        Ok((pipeline[0], pipeline_layout))
    }

    fn record_command_buffer(&self, cmd_buffer: vk::CommandBuffer, image_idx: u32) -> VulkanResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .describe_err("Failed to begin recording command buffer")?;
        }

        let clear_color = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let renderpass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_idx as usize])
            .render_area(self.swapchain.extent_rect())
            .clear_values(&clear_color);

        unsafe {
            self.device
                .cmd_begin_render_pass(cmd_buffer, &renderpass_info, vk::SubpassContents::INLINE);
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device
                .cmd_bind_vertex_buffers(cmd_buffer, 0, array::from_ref(&self.vertex_buffer), &[0]);
            self.device
                .cmd_bind_index_buffer(cmd_buffer, self.index_buffer, 0, vk::IndexType::UINT16);
            self.device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &self.descriptor_sets[self.current_frame..=self.current_frame],
                &[],
            );
            self.device
                .cmd_set_viewport(cmd_buffer, 0, array::from_ref(&self.swapchain.viewport()));
            self.device
                .cmd_set_scissor(cmd_buffer, 0, array::from_ref(&self.swapchain.extent_rect()));
            self.device.cmd_draw_indexed(cmd_buffer, 6, 1, 0, 0, 0);
            self.device.cmd_end_render_pass(cmd_buffer);
            self.device
                .end_command_buffer(cmd_buffer)
                .describe_err("Failed to end recording command buffer")?;
        }

        Ok(())
    }

    pub fn draw_frame(&mut self, window: &Window) -> VulkanResult<()> {
        let in_flight_fen = self.sync[self.current_frame].in_flight_fen;
        let image_avail_sem = self.sync[self.current_frame].image_avail_sem;
        let render_finish_sem = self.sync[self.current_frame].render_finished_sem;
        let command_buffer = self.command_buffers[self.current_frame];

        self.update_uniforms();

        let image_idx = unsafe {
            self.device
                .wait_for_fences(array::from_ref(&in_flight_fen), true, u64::MAX)
                .describe_err("Failed waiting for error")?;
            let acquire_res =
                self.device
                    .swapchain_utils
                    .acquire_next_image(self.swapchain.handle, u64::MAX, image_avail_sem, vk::Fence::null());
            match acquire_res {
                Ok((idx, _)) => idx,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    eprintln!("swapchain out of date");
                    self.recreate_swapchain(window)?;
                    return Ok(());
                }
                Err(e) => return Err(VkError::VulkanMsg("Failed to acquire swapchain image", e)),
            }
        };

        unsafe {
            self.device
                .reset_fences(array::from_ref(&in_flight_fen))
                .describe_err("Failed resetting fences")?;
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .describe_err("Failed to reset command buffer")?;
        }

        self.record_command_buffer(command_buffer, image_idx)?;

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(array::from_ref(&image_avail_sem))
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(array::from_ref(&command_buffer))
            .signal_semaphores(array::from_ref(&render_finish_sem))
            .build();

        unsafe {
            self.device
                .queue_submit(self.device.graphics_queue, array::from_ref(&submit_info), in_flight_fen)
                .describe_err("Failed to submit draw command buffer")?
        }

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(array::from_ref(&render_finish_sem))
            .swapchains(array::from_ref(&self.swapchain.handle))
            .image_indices(array::from_ref(&image_idx));

        let suboptimal = unsafe {
            self.device
                .swapchain_utils
                .queue_present(self.device.present_queue, &present_info)
                .describe_err("Failed to present queue")?
        };

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        if suboptimal {
            eprintln!("swapchain suboptimal");
            self.recreate_swapchain(window)?;
        }

        Ok(())
    }

    fn update_uniforms(&mut self) {
        let time = ((Instant::now() - self.start_time).as_micros() as f64 / 1000000.0) as f32;
        let aspect = self.swapchain.extent.width as f32 / self.swapchain.extent.height as f32;
        let ubo = UniformBufferObject {
            model: Matrix4::from_axis_angle(Vector3::unit_z(), Deg(time * 90.0)),
            view: Matrix4::look_at_rh(Point3::new(2.0, 2.0, 2.0), Point3::new(0.0, 0.0, 0.0), -Vector3::unit_z()),
            proj: cgmath::perspective(Deg(45.0), aspect, 0.1, 10.0),
        };
        self.uniforms[self.current_frame].write_uniforms(ubo);
    }

    unsafe fn cleanup_swapchain(&mut self) {
        for &fb in &self.framebuffers {
            self.device.destroy_framebuffer(fb, None);
        }
        self.swapchain.cleanup(&self.device);
    }

    fn recreate_swapchain(&mut self, window: &Window) -> VulkanResult<()> {
        unsafe { self.device.device_wait_idle()? };
        self.device.update_surface_info()?;
        let swapchain = self
            .device
            .create_swapchain(window, SWAPCHAIN_IMAGE_COUNT, Some(self.swapchain.handle))?;
        let framebuffers = self.device.create_framebuffers(&swapchain, self.render_pass)?;
        unsafe { self.cleanup_swapchain() };
        self.swapchain = swapchain;
        self.framebuffers = framebuffers;

        Ok(())
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vb_memory, None);
            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.ib_memory, None);
            for elem in &mut self.sync {
                elem.cleanup(&self.device);
            }
            for elem in &mut self.uniforms {
                elem.cleanup(&self.device)
            }
            self.device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.cleanup_swapchain();
        }
    }
}

struct FrameSyncState {
    image_avail_sem: vk::Semaphore,
    render_finished_sem: vk::Semaphore,
    in_flight_fen: vk::Fence,
}

impl FrameSyncState {
    fn new(device: &VulkanDevice) -> VulkanResult<Self> {
        Ok(Self {
            image_avail_sem: device.create_semaphore()?,
            render_finished_sem: device.create_semaphore()?,
            in_flight_fen: device.create_fence()?,
        })
    }

    unsafe fn cleanup(&mut self, device: &ash::Device) {
        device.destroy_semaphore(self.image_avail_sem, None);
        device.destroy_semaphore(self.render_finished_sem, None);
        device.destroy_fence(self.in_flight_fen, None);
    }
}

struct UniformData {
    uniform_buffer: vk::Buffer,
    ub_memory: vk::DeviceMemory,
    ub_mapped: *mut UniformBufferObject,
}

impl UniformData {
    fn new(device: &VulkanDevice) -> VulkanResult<Self> {
        let size = std::mem::size_of::<UniformBufferObject>() as _;
        let (uniform_buffer, ub_memory) = device.allocate_buffer(
            size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let ub_mapped = unsafe { device.map_memory(ub_memory, 0, size, Default::default())? as _ };
        Ok(Self {
            uniform_buffer,
            ub_memory,
            ub_mapped,
        })
    }

    fn write_uniforms(&mut self, ubo: UniformBufferObject) {
        unsafe {
            std::ptr::write_volatile(self.ub_mapped, ubo);
        }
    }

    unsafe fn cleanup(&mut self, device: &ash::Device) {
        device.destroy_buffer(self.uniform_buffer, None);
        device.free_memory(self.ub_memory, None);
        self.ub_mapped = std::ptr::null_mut();
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}
