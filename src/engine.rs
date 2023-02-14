use crate::device::{SwapchainInfo, VkBuffer, VkImage, VulkanDevice};
use crate::types::*;
use ash::vk;
use cgmath::{Deg, Matrix4, Point3, Vector3};
use cstr::cstr;
use inline_spirv::include_spirv;
use std::array;
use std::time::Instant;

const SWAPCHAIN_IMAGE_COUNT: u32 = 3;
const MAX_FRAMES_IN_FLIGHT: usize = 2;
//type Vertex = ([f32; 3], [f32; 3], [f32; 2]);
type Vertex = obj::TexturedVertex;

pub struct VulkanEngine {
    device: VulkanDevice,
    window_size: WinSize,
    window_resized: bool,
    swapchain: SwapchainInfo,
    depth_image: VkImage,
    depth_imgview: vk::ImageView,
    depth_format: vk::Format,
    framebuffers: Vec<vk::Framebuffer>,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    frame_state: Vec<FrameState>,
    current_frame: usize,
    vertex_buffer: VkBuffer,
    index_buffer: VkBuffer,
    index_count: u32,
    texture: VkImage,
    tex_imgview: vk::ImageView,
    tex_sampler: vk::Sampler,
    start_time: Instant,
    frame_time: Instant,
}

impl VulkanEngine {
    pub fn new(
        vk: VulkanDevice, window_size: WinSize, vertices: &[Vertex], indices: &[u32], img_width: u32, img_height: u32, img_data: &[u8],
    ) -> VulkanResult<Self> {
        let vert_spv = include_spirv!("src/shaders/texture.vert.glsl", vert, glsl);
        let frag_spv = include_spirv!("src/shaders/texture.frag.glsl", frag, glsl);

        let swapchain = vk.create_swapchain(window_size, SWAPCHAIN_IMAGE_COUNT, None)?;
        let (depth_image, depth_format) = vk.create_depth_image(swapchain.extent.width, swapchain.extent.height, None)?;
        let depth_imgview = vk.create_image_view(*depth_image, depth_format, vk::ImageAspectFlags::DEPTH)?;
        let render_pass = Self::create_render_pass(&vk, swapchain.format, depth_format)?;
        let framebuffers = vk.create_framebuffers(&swapchain, render_pass, depth_imgview)?;

        let command_buffers = vk.create_command_buffers(MAX_FRAMES_IN_FLIGHT as u32)?;
        let frame_state = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| FrameState::new(&vk, command_buffers[i]))
            .collect::<Result<Vec<_>, _>>()?;

        let texture = vk.create_texture(img_width, img_height, img_data)?;
        let tex_imgview = vk.create_image_view(*texture, vk::Format::R8G8B8A8_SRGB, vk::ImageAspectFlags::COLOR)?;
        let tex_sampler = vk.create_texture_sampler(vk::SamplerAddressMode::REPEAT)?;

        let descriptor_layout = Self::create_descriptor_set_layout(&vk)?;
        let descriptor_pool = Self::create_descriptor_pool(&vk, MAX_FRAMES_IN_FLIGHT as u32)?;
        let descriptor_sets = Self::create_descriptor_sets(
            &vk,
            descriptor_pool,
            &[descriptor_layout; MAX_FRAMES_IN_FLIGHT],
            &frame_state,
            tex_imgview,
            tex_sampler,
        )?;
        let pipeline_layout = Self::create_pipeline_layout(&vk, descriptor_layout)?;
        let pipeline = Self::create_graphics_pipeline(
            &vk,
            vert_spv,
            frag_spv,
            &[Vertex::binding_desc(0)],
            &Vertex::attr_desc(0),
            render_pass,
            pipeline_layout,
        )?;

        let vertex_buffer = vk.create_buffer(vertices, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let index_buffer = vk.create_buffer(indices, vk::BufferUsageFlags::INDEX_BUFFER)?;

        let now = Instant::now();

        Ok(Self {
            device: vk,
            window_size,
            window_resized: false,
            swapchain,
            depth_image,
            depth_imgview,
            depth_format,
            framebuffers,
            render_pass,
            pipeline,
            pipeline_layout,
            descriptor_layout,
            descriptor_pool,
            descriptor_sets,
            frame_state,
            current_frame: 0,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as _,
            texture,
            tex_imgview,
            tex_sampler,
            start_time: now,
            frame_time: now,
        })
    }

    pub fn resize(&mut self, window_size: WinSize) {
        self.window_size = window_size;
        self.window_resized = true;
    }

    pub fn get_frame_time(&self) -> Instant {
        self.frame_time
    }

    fn create_render_pass(device: &VulkanDevice, color_format: vk::Format, depth_format: vk::Format) -> VulkanResult<vk::RenderPass> {
        eprintln!("color_format: {color_format:?}, depth_format: {depth_format:?}");
        let attachments = [
            vk::AttachmentDescription {
                flags: Default::default(),
                format: color_format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            },
            vk::AttachmentDescription {
                flags: Default::default(),
                format: depth_format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::DONT_CARE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            },
        ];

        let color_attach_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attach_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(array::from_ref(&color_attach_ref))
            .depth_stencil_attachment(&depth_attach_ref)
            .build();

        let dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .build();

        let render_pass_ci = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(array::from_ref(&subpass))
            .dependencies(array::from_ref(&dependency));

        unsafe {
            device
                .create_render_pass(&render_pass_ci, None)
                .describe_err("Failed to create render pass")
        }
    }

    fn create_descriptor_pool(device: &VulkanDevice, max_count: u32) -> VulkanResult<vk::DescriptorPool> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: max_count,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: max_count,
            },
        ];
        let pool_ci = vk::DescriptorPoolCreateInfo::builder().pool_sizes(&pool_sizes).max_sets(max_count);
        unsafe {
            device
                .create_descriptor_pool(&pool_ci, None)
                .describe_err("Failed to create descriptor pool")
        }
    }

    fn create_descriptor_set_layout(device: &VulkanDevice) -> VulkanResult<vk::DescriptorSetLayout> {
        let layout_bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
        ];

        let desc_layout_ci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);

        unsafe {
            device
                .create_descriptor_set_layout(&desc_layout_ci, None)
                .describe_err("Failed to create descriptor set layout")
        }
    }

    fn create_descriptor_sets(
        device: &VulkanDevice, pool: vk::DescriptorPool, layouts: &[vk::DescriptorSetLayout], frame_state: &[FrameState],
        image_view: vk::ImageView, sampler: vk::Sampler,
    ) -> VulkanResult<Vec<vk::DescriptorSet>> {
        assert_eq!(layouts.len(), frame_state.len());
        let alloc_info = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(pool).set_layouts(&layouts);
        let desc_sets = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .describe_err("Failed to allocate descriptor sets")?
        };

        for (&desc_set, fstate) in desc_sets.iter().zip(frame_state) {
            let buffer_info = vk::DescriptorBufferInfo {
                buffer: *fstate.uniforms.buffer,
                offset: 0,
                range: std::mem::size_of::<UniformBufferObject>() as _,
            };
            let image_info = vk::DescriptorImageInfo {
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image_view,
                sampler,
            };
            let descr_write = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(desc_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(array::from_ref(&buffer_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(desc_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(array::from_ref(&image_info))
                    .build(),
            ];
            unsafe {
                device.update_descriptor_sets(&descr_write, &[]);
            }
        }

        Ok(desc_sets)
    }

    fn create_pipeline_layout(device: &VulkanDevice, desc_set_layout: vk::DescriptorSetLayout) -> VulkanResult<vk::PipelineLayout> {
        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::builder().set_layouts(array::from_ref(&desc_set_layout));

        unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_ci, None)
                .describe_err("Failed to create pipeline layout")
        }
    }

    fn create_graphics_pipeline(
        device: &VulkanDevice, vert_shader_spv: &[u32], frag_shader_spv: &[u32], binding_desc: &[vk::VertexInputBindingDescription],
        attr_desc: &[vk::VertexInputAttributeDescription], render_pass: vk::RenderPass, pipeline_layout: vk::PipelineLayout,
    ) -> VulkanResult<vk::Pipeline> {
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

        let vertex_input_ci = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(binding_desc)
            .vertex_attribute_descriptions(attr_desc);

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

        let depth_stencil_ci = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);

        let pipeline_ci = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages_ci)
            .vertex_input_state(&vertex_input_ci)
            .input_assembly_state(&input_assembly_ci)
            .viewport_state(&viewport_state_ci)
            .rasterization_state(&rasterizer_ci)
            .multisample_state(&multisample_ci)
            .color_blend_state(&color_blend_ci)
            .depth_stencil_state(&depth_stencil_ci)
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

        Ok(pipeline[0])
    }

    fn record_command_buffer(&self, cmd_buffer: vk::CommandBuffer, image_idx: u32) -> VulkanResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .describe_err("Failed to begin recording command buffer")?;
        }

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            },
        ];
        let renderpass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_idx as usize])
            .render_area(self.swapchain.extent_rect())
            .clear_values(&clear_values);

        unsafe {
            self.device
                .cmd_begin_render_pass(cmd_buffer, &renderpass_info, vk::SubpassContents::INLINE);
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device
                .cmd_bind_vertex_buffers(cmd_buffer, 0, array::from_ref(&*self.vertex_buffer), &[0]);
            self.device
                .cmd_bind_index_buffer(cmd_buffer, *self.index_buffer, 0, vk::IndexType::UINT32);
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
            self.device.cmd_draw_indexed(cmd_buffer, self.index_count, 1, 0, 0, 0);
            self.device.cmd_end_render_pass(cmd_buffer);
            self.device
                .end_command_buffer(cmd_buffer)
                .describe_err("Failed to end recording command buffer")?;
        }

        Ok(())
    }

    pub fn draw_frame(&mut self) -> VulkanResult<bool> {
        let frame = &self.frame_state[self.current_frame];
        let in_flight_fen = frame.in_flight_fen;
        let image_avail_sem = frame.image_avail_sem;
        let render_finish_sem = frame.render_finished_sem;
        let command_buffer = frame.command_buffer;

        let image_idx = unsafe {
            self.device
                .wait_for_fences(array::from_ref(&in_flight_fen), true, u64::MAX)
                .describe_err("Failed waiting for fence")?;
            self.frame_time = Instant::now();
            let acquire_res = self
                .device
                .swapchain_utils
                .acquire_next_image(*self.swapchain, u64::MAX, image_avail_sem, vk::Fence::null());
            match acquire_res {
                Ok((idx, _)) => idx,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    eprintln!("swapchain out of date");
                    self.recreate_swapchain()?;
                    return Ok(false);
                }
                Err(e) => return Err(VkError::VulkanMsg("Failed to acquire swapchain image", e)),
            }
        };

        self.update_uniforms();

        unsafe {
            self.device
                .reset_fences(array::from_ref(&in_flight_fen))
                .describe_err("Failed resetting fences")?;
            self.device
                .reset_command_buffer(command_buffer, Default::default())
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
            .swapchains(array::from_ref(&*self.swapchain))
            .image_indices(array::from_ref(&image_idx));

        let suboptimal = unsafe {
            self.device
                .swapchain_utils
                .queue_present(self.device.present_queue, &present_info)
                .describe_err("Failed to present queue")?
        };

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        if suboptimal || self.window_resized {
            eprintln!("swapchain suboptimal");
            self.recreate_swapchain()?;
            self.window_resized = false;
        }

        Ok(true)
    }

    fn update_uniforms(&mut self) {
        let time = ((self.frame_time - self.start_time).as_micros() as f64 / 1000000.0) as f32;
        let aspect = self.swapchain.extent.width as f32 / self.swapchain.extent.height as f32;
        let ubo = UniformBufferObject {
            model: Matrix4::from_axis_angle(Vector3::unit_z(), Deg(time * 90.0)),
            view: Matrix4::look_at_rh(Point3::new(2.0, 2.0, 2.0), Point3::new(0.0, 0.0, 0.0), -Vector3::unit_z()),
            proj: cgmath::perspective(Deg(45.0), aspect, 0.1, 10.0),
        };
        self.frame_state[self.current_frame].uniforms.write_uniforms(ubo);
    }

    unsafe fn cleanup_swapchain(&mut self) {
        for &fb in &self.framebuffers {
            self.device.destroy_framebuffer(fb, None);
        }
        self.device.destroy_image_view(self.depth_imgview, None);
        self.depth_image.cleanup(&self.device);
        self.swapchain.cleanup(&self.device);
    }

    fn recreate_swapchain(&mut self) -> VulkanResult<()> {
        unsafe { self.device.device_wait_idle()? };
        self.device.update_surface_info()?;
        let swapchain = self
            .device
            .create_swapchain(self.window_size, SWAPCHAIN_IMAGE_COUNT, Some(*self.swapchain))?;
        let (image, format) = self
            .device
            .create_depth_image(swapchain.extent.width, swapchain.extent.height, Some(self.depth_format))?;
        let imgview = self.device.create_image_view(*image, format, vk::ImageAspectFlags::DEPTH)?;
        let framebuffers = self.device.create_framebuffers(&swapchain, self.render_pass, imgview)?;
        unsafe { self.cleanup_swapchain() };
        self.swapchain = swapchain;
        self.framebuffers = framebuffers;
        self.depth_image = image;
        self.depth_imgview = imgview;

        Ok(())
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.vertex_buffer.cleanup(&self.device);
            self.index_buffer.cleanup(&self.device);
            self.device.destroy_sampler(self.tex_sampler, None);
            self.device.destroy_image_view(self.tex_imgview, None);
            self.texture.cleanup(&self.device);
            self.frame_state.cleanup(&self.device);
            self.device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.cleanup_swapchain();
        }
    }
}

struct FrameState {
    image_avail_sem: vk::Semaphore,
    render_finished_sem: vk::Semaphore,
    in_flight_fen: vk::Fence,
    uniforms: UniformData<UniformBufferObject>,
    command_buffer: vk::CommandBuffer,
}

impl FrameState {
    fn new(device: &VulkanDevice, command_buffer: vk::CommandBuffer) -> VulkanResult<Self> {
        Ok(Self {
            image_avail_sem: device.create_semaphore()?,
            render_finished_sem: device.create_semaphore()?,
            in_flight_fen: device.create_fence()?,
            uniforms: UniformData::new(device)?,
            command_buffer,
        })
    }
}

impl Cleanup<ash::Device> for FrameState {
    unsafe fn cleanup(&mut self, device: &ash::Device) {
        device.destroy_semaphore(self.image_avail_sem, None);
        device.destroy_semaphore(self.render_finished_sem, None);
        device.destroy_fence(self.in_flight_fen, None);
        self.uniforms.cleanup(device);
    }
}

struct UniformData<T> {
    buffer: VkBuffer,
    ub_mapped: *mut T,
}

impl<T> UniformData<T> {
    fn new(device: &VulkanDevice) -> VulkanResult<Self> {
        let size = std::mem::size_of::<T>() as _;
        let buffer = device.allocate_buffer(
            size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let ub_mapped = unsafe { device.map_memory(buffer.memory, 0, size, Default::default())? as _ };
        Ok(Self { buffer, ub_mapped })
    }

    fn write_uniforms(&mut self, ubo: T) {
        unsafe {
            std::ptr::write_volatile(self.ub_mapped, ubo);
        }
    }
}

impl<T> Cleanup<ash::Device> for UniformData<T> {
    unsafe fn cleanup(&mut self, device: &ash::Device) {
        device.unmap_memory(self.buffer.memory);
        self.ub_mapped = std::ptr::null_mut();
        self.buffer.cleanup(device);
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}
