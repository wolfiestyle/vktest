use crate::camera::Camera;
use crate::device::{ImageData, Swapchain, UniformBuffer, VkBuffer, VkImage, VulkanDevice};
use crate::types::*;
use ash::vk;
use cstr::cstr;
use glam::{Affine3A, Mat4};
use inline_spirv::include_spirv;
use std::slice;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SWAPCHAIN_IMAGE_COUNT: u32 = 3;
const MAX_FRAMES_IN_FLIGHT: usize = 2;
//type Vertex = ([f32; 3], [f32; 3], [f32; 2]);
type Vertex = obj::TexturedVertex;

pub struct VulkanEngine {
    pub device: Arc<VulkanDevice>,
    window_size: WinSize,
    window_resized: bool,
    pub(crate) swapchain: Swapchain,
    shader: Shader,
    desc_layout: vk::DescriptorSetLayout,
    pipeline: Pipeline,
    pipeline_layout: vk::PipelineLayout,
    bg_shader: Shader,
    bg_pipeline: Pipeline,
    bg_texture: Texture,
    main_cmd_pool: vk::CommandPool,
    secondary_cmd_pool: vk::CommandPool,
    main_cmd_buffers: Vec<vk::CommandBuffer>,
    frame_state: Vec<FrameState>,
    current_frame: usize,
    vertex_buffer: VkBuffer,
    index_buffer: VkBuffer,
    index_count: u32,
    texture: Texture,
    tex_sampler: vk::Sampler,
    prev_frame_time: Instant,
    last_frame_time: Instant,
    pub camera: Camera,
    pub model: Affine3A,
}

impl VulkanEngine {
    pub fn new(
        vk: VulkanDevice, window_size: WinSize, vertices: &[Vertex], indices: &[u32], img_dims: (u32, u32), img_data: &[u8],
        skybox_dims: (u32, u32), skybox_data: &[&[u8]; 6],
    ) -> VulkanResult<Self> {
        let depth_format = vk.find_depth_format(false)?;
        let swapchain = vk.create_swapchain(window_size, SWAPCHAIN_IMAGE_COUNT, depth_format)?;
        eprintln!("color_format: {:?}, depth_format: {depth_format:?}", swapchain.format);

        let main_cmd_pool = vk.create_command_pool(vk.dev_info.graphics_idx, vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)?;
        let secondary_cmd_pool = vk.create_command_pool(vk.dev_info.graphics_idx, vk::CommandPoolCreateFlags::TRANSIENT)?;
        let main_cmd_buffers = vk.create_command_buffers(main_cmd_pool, MAX_FRAMES_IN_FLIGHT as u32, vk::CommandBufferLevel::PRIMARY)?;
        let frame_state = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| FrameState::new(&vk))
            .collect::<Result<Vec<_>, _>>()?;

        let tex_sampler = vk.create_texture_sampler(vk::Filter::LINEAR, vk::Filter::LINEAR, vk::SamplerAddressMode::REPEAT)?;
        let texture = Texture::new(&vk, img_dims.0, img_dims.1, vk::Format::R8G8B8A8_SRGB, img_data, tex_sampler)?;

        let shader = Shader::new(
            &vk,
            include_spirv!("src/shaders/texture.vert.glsl", vert, glsl),
            include_spirv!("src/shaders/texture.frag.glsl", frag, glsl),
        )?;
        let desc_layout = vk.create_descriptor_set_layout(&[
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
        ])?;
        let pipeline_layout = vk.create_pipeline_layout(slice::from_ref(&desc_layout), &[])?;
        let pipeline = Pipeline::new::<Vertex>(&vk, &shader, pipeline_layout, &swapchain, PipelineMode::Opaque)?;

        let bg_shader = Shader::new(
            &vk,
            include_spirv!("src/shaders/skybox.vert.glsl", vert, glsl),
            include_spirv!("src/shaders/skybox.frag.glsl", frag, glsl),
        )?;
        let bg_pipeline = Pipeline::new_no_input(&vk, &bg_shader, pipeline_layout, &swapchain, PipelineMode::Background)?;
        let bg_texture = Texture::new_cubemap(
            &vk,
            skybox_dims.0,
            skybox_dims.1,
            vk::Format::R8G8B8A8_SRGB,
            skybox_data,
            tex_sampler,
        )?;

        let vertex_buffer = vk.create_buffer_from_data(vertices, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let index_buffer = vk.create_buffer_from_data(indices, vk::BufferUsageFlags::INDEX_BUFFER)?;

        let camera = Camera::default();
        let now = Instant::now();

        Ok(Self {
            device: vk.into(),
            window_size,
            window_resized: false,
            swapchain,
            shader,
            desc_layout,
            pipeline,
            pipeline_layout,
            bg_shader,
            bg_pipeline,
            bg_texture,
            main_cmd_pool,
            secondary_cmd_pool,
            main_cmd_buffers,
            frame_state,
            current_frame: 0,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as _,
            texture,
            tex_sampler,
            prev_frame_time: now,
            last_frame_time: now,
            camera,
            model: Affine3A::IDENTITY,
        })
    }

    pub fn resize(&mut self, window_size: impl Into<WinSize>) {
        let window_size = window_size.into();
        eprintln!("window size: {} x {}", window_size.width, window_size.height);
        if window_size != self.window_size {
            self.window_size = window_size;
            self.window_resized = true;
        }
    }

    pub fn get_frame_timestamp(&self) -> Instant {
        self.last_frame_time
    }

    pub fn get_frame_time(&self) -> Duration {
        self.last_frame_time - self.prev_frame_time
    }

    fn record_primary_command_buffer(
        &self, cmd_buffer: vk::CommandBuffer, image_idx: usize, secondaries: &[vk::CommandBuffer],
    ) -> VulkanResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .describe_err("Failed to begin recording command buffer")?;
        }

        let color_attach = vk::RenderingAttachmentInfo::builder()
            .image_view(self.swapchain.image_views[image_idx])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            });
        let depth_attach = vk::RenderingAttachmentInfo::builder()
            .image_view(self.swapchain.depth_imgviews[image_idx])
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            });
        let render_info = vk::RenderingInfo::builder()
            .flags(vk::RenderingFlags::CONTENTS_SECONDARY_COMMAND_BUFFERS)
            .render_area(self.swapchain.extent_rect())
            .layer_count(1)
            .color_attachments(slice::from_ref(&color_attach))
            .depth_attachment(&depth_attach);

        unsafe {
            self.device.transition_image_layout(
                cmd_buffer,
                self.swapchain.images[image_idx],
                self.swapchain.format,
                1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
            self.device.dynrender_fn.cmd_begin_rendering(cmd_buffer, &render_info);
            self.device.cmd_execute_commands(cmd_buffer, secondaries);
            self.device.dynrender_fn.cmd_end_rendering(cmd_buffer);
            self.device.transition_image_layout(
                cmd_buffer,
                self.swapchain.images[image_idx],
                self.swapchain.format,
                1,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
            self.device
                .end_command_buffer(cmd_buffer)
                .describe_err("Failed to end recording command buffer")?;
        }
        Ok(())
    }

    pub fn begin_secondary_draw_commands(&self) -> VulkanResult<vk::CommandBuffer> {
        let cmd_buffer = self
            .device
            .create_command_buffers(self.secondary_cmd_pool, 1, vk::CommandBufferLevel::SECONDARY)?[0];
        let mut render_info = vk::CommandBufferInheritanceRenderingInfo::builder()
            .color_attachment_formats(slice::from_ref(&self.swapchain.format))
            .depth_attachment_format(self.swapchain.depth_format)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let inherit_info = vk::CommandBufferInheritanceInfo::builder().push_next(&mut render_info);
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inherit_info);
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .describe_err("Failed to begin recording secondary command buffer")?;
        }
        Ok(cmd_buffer)
    }

    pub fn end_secondary_draw_commands(&self, cmd_buffer: vk::CommandBuffer) -> VulkanResult<vk::CommandBuffer> {
        unsafe {
            self.device
                .end_command_buffer(cmd_buffer)
                .describe_err("Failed to end recording secodary command buffer")?;
        }
        Ok(cmd_buffer)
    }

    pub fn draw_object(&self) -> VulkanResult<vk::CommandBuffer> {
        let cmd_buffer = self.begin_secondary_draw_commands()?;

        let buffer_info = self.frame_state[self.current_frame].uniforms.descriptor();
        let image_info = self.texture.descriptor();
        unsafe {
            // object
            self.device
                .debug(|d| d.cmd_begin_label(cmd_buffer, "3D object", [0.2, 0.6, 0.4, 1.0]));
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle);
            self.device
                .cmd_set_viewport(cmd_buffer, 0, slice::from_ref(&self.swapchain.viewport()));
            self.device
                .cmd_set_scissor(cmd_buffer, 0, slice::from_ref(&self.swapchain.extent_rect()));
            self.device.pushdesc_fn.cmd_push_descriptor_set(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(slice::from_ref(&buffer_info))
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(slice::from_ref(&image_info))
                        .build(),
                ],
            );
            self.device
                .cmd_bind_vertex_buffers(cmd_buffer, 0, slice::from_ref(&*self.vertex_buffer), &[0]);
            self.device
                .cmd_bind_index_buffer(cmd_buffer, *self.index_buffer, 0, vk::IndexType::UINT32);
            self.device.cmd_draw_indexed(cmd_buffer, self.index_count, 1, 0, 0, 0);
            self.device.debug(|d| d.cmd_end_label(cmd_buffer));
        }

        let image_info = self.bg_texture.descriptor();
        unsafe {
            // background
            self.device
                .debug(|d| d.cmd_begin_label(cmd_buffer, "Background", [0.2, 0.4, 0.6, 1.0]));
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.bg_pipeline.handle);
            self.device.pushdesc_fn.cmd_push_descriptor_set(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[vk::WriteDescriptorSet::builder()
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(slice::from_ref(&image_info))
                    .build()],
            );
            self.device.cmd_draw(cmd_buffer, 6, 1, 0, 0);
            self.device.debug(|d| d.cmd_end_label(cmd_buffer));
        }

        self.end_secondary_draw_commands(cmd_buffer)
    }

    pub fn submit_draw_commands(&mut self, draw_commands: &[vk::CommandBuffer]) -> VulkanResult<bool> {
        let frame = &mut self.frame_state[self.current_frame];
        let in_flight_fen = frame.in_flight_fen;
        let image_avail_sem = frame.image_avail_sem;
        let render_finish_sem = frame.render_finished_sem;
        let command_buffer = self.main_cmd_buffers[self.current_frame];

        unsafe {
            self.device
                .wait_for_fences(slice::from_ref(&in_flight_fen), true, u64::MAX)
                .describe_err("Failed waiting for fence")?;
        }
        // all previous work for this frame is done at this point
        if !frame.cmd_buffers.is_empty() {
            unsafe { self.device.free_command_buffers(self.secondary_cmd_pool, &frame.cmd_buffers) };
            frame.cmd_buffers.clear();
        }
        self.prev_frame_time = self.last_frame_time;
        self.last_frame_time = Instant::now();
        frame.cmd_buffers.extend(draw_commands);
        self.update_uniforms()?;

        let image_idx = unsafe {
            let acquire_res = self
                .device
                .swapchain_fn
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

        unsafe {
            self.device
                .reset_fences(slice::from_ref(&in_flight_fen))
                .describe_err("Failed resetting fences")?;
            self.device
                .reset_command_buffer(command_buffer, Default::default())
                .describe_err("Failed to reset command buffer")?;
        }

        self.record_primary_command_buffer(command_buffer, image_idx as _, draw_commands)?;

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(slice::from_ref(&image_avail_sem))
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(slice::from_ref(&command_buffer))
            .signal_semaphores(slice::from_ref(&render_finish_sem));
        unsafe {
            self.device
                .queue_submit(self.device.graphics_queue, slice::from_ref(&submit_info), in_flight_fen)
                .describe_err("Failed to submit draw command buffer")?
        }

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(slice::from_ref(&render_finish_sem))
            .swapchains(slice::from_ref(&*self.swapchain))
            .image_indices(slice::from_ref(&image_idx));
        let suboptimal = unsafe {
            self.device
                .swapchain_fn
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

    fn update_uniforms(&mut self) -> VulkanResult<()> {
        let view = self.camera.get_view_transform();
        let proj = self.camera.get_projection(self.swapchain.aspect());
        let viewproj = proj * view;
        let viewproj_inv = viewproj.inverse();
        let ubo = UniformBufferObject {
            mvp: viewproj * self.model,
            viewproj_inv,
        };
        self.frame_state[self.current_frame].uniforms.write_uniforms(&self.device, ubo)
    }

    fn recreate_swapchain(&mut self) -> VulkanResult<()> {
        let new_swapchain = self.device.recreate_swapchain(self.window_size, &self.swapchain)?;
        unsafe {
            self.device.device_wait_idle()?;
            self.swapchain.cleanup(&self.device);
        }
        self.swapchain = new_swapchain;
        Ok(())
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.vertex_buffer.cleanup(&self.device);
            self.index_buffer.cleanup(&self.device);
            self.texture.cleanup(&self.device);
            self.device.destroy_sampler(self.tex_sampler, None);
            self.bg_texture.cleanup(&self.device);
            self.frame_state.cleanup(&self.device);
            self.device.destroy_command_pool(self.main_cmd_pool, None);
            self.device.destroy_command_pool(self.secondary_cmd_pool, None);
            self.pipeline.cleanup(&self.device);
            self.bg_pipeline.cleanup(&self.device);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.desc_layout, None);
            self.shader.cleanup(&self.device);
            self.bg_shader.cleanup(&self.device);
            self.swapchain.cleanup(&self.device);
        }
    }
}

pub struct Pipeline {
    handle: vk::Pipeline,
}

impl Pipeline {
    pub fn new<Vert: VertexInput>(
        device: &VulkanDevice, shader: &Shader, layout: vk::PipelineLayout, swapchain: &Swapchain, mode: PipelineMode,
    ) -> VulkanResult<Self> {
        let binding_desc = Vert::binding_desc(0);
        let attr_desc = Vert::attr_desc(0);
        let pipeline = Self::create_pipeline(device, shader, layout, swapchain, slice::from_ref(&binding_desc), &attr_desc, mode)?;
        device.debug(|d| d.set_object_name(device, &pipeline.handle, &format!("Pipeline<{}>", std::any::type_name::<Vert>())));
        Ok(pipeline)
    }

    pub fn new_no_input(
        device: &VulkanDevice, shader: &Shader, layout: vk::PipelineLayout, swapchain: &Swapchain, mode: PipelineMode,
    ) -> VulkanResult<Self> {
        let pipeline = Pipeline::create_pipeline(device, shader, layout, swapchain, &[], &[], mode)?;
        device.debug(|d| d.set_object_name(device, &pipeline.handle, "Pipeline<()>"));
        Ok(pipeline)
    }

    fn create_pipeline(
        device: &VulkanDevice, shader: &Shader, layout: vk::PipelineLayout, swapchain: &Swapchain,
        binding_desc: &[vk::VertexInputBindingDescription], attr_desc: &[vk::VertexInputAttributeDescription], mode: PipelineMode,
    ) -> VulkanResult<Self> {
        let entry_point = cstr!("main");
        let shader_stages_ci = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader.vert)
                .name(entry_point)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader.frag)
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
            .cull_mode(mode.cull_mode())
            .front_face(vk::FrontFace::CLOCKWISE);

        let multisample_ci = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0);

        let color_attach = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(mode.blend_enable())
            .src_color_blend_factor(vk::BlendFactor::ONE) // premultiplied alpha
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blend_ci = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(slice::from_ref(&color_attach));

        let depth_stencil_ci = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(mode.depth_test())
            .depth_write_enable(mode.depth_write())
            .depth_compare_op(mode.depth_compare_op())
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);

        let mut pipeline_rendering_ci = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(slice::from_ref(&swapchain.format))
            .depth_attachment_format(swapchain.depth_format);

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
            .layout(layout)
            .push_next(&mut pipeline_rendering_ci);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), slice::from_ref(&pipeline_ci), None)
                .map_err(|(_, err)| VkError::VulkanMsg("Error creating pipeline", err))?
        };

        Ok(Self { handle: pipeline[0] })
    }
}

impl Cleanup<VulkanDevice> for Pipeline {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_pipeline(self.handle, None);
    }
}

impl std::ops::Deref for Pipeline {
    type Target = vk::Pipeline;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PipelineMode {
    Opaque,
    Transparent,
    Background,
    Overlay,
}

impl PipelineMode {
    fn depth_test(self) -> bool {
        match self {
            Self::Overlay => false,
            _ => true,
        }
    }

    fn depth_write(self) -> bool {
        match self {
            Self::Opaque => true,
            _ => false,
        }
    }

    fn depth_compare_op(self) -> vk::CompareOp {
        match self {
            Self::Opaque | Self::Transparent => vk::CompareOp::LESS,
            Self::Background => vk::CompareOp::EQUAL,
            Self::Overlay => vk::CompareOp::ALWAYS,
        }
    }

    fn cull_mode(self) -> vk::CullModeFlags {
        match self {
            Self::Overlay => vk::CullModeFlags::NONE,
            _ => vk::CullModeFlags::BACK,
        }
    }

    fn blend_enable(self) -> bool {
        match self {
            Self::Transparent | Self::Overlay => true,
            _ => false,
        }
    }
}

struct FrameState {
    image_avail_sem: vk::Semaphore,
    render_finished_sem: vk::Semaphore,
    in_flight_fen: vk::Fence,
    uniforms: UniformBuffer<UniformBufferObject>,
    cmd_buffers: Vec<vk::CommandBuffer>,
}

impl FrameState {
    fn new(device: &VulkanDevice) -> VulkanResult<Self> {
        let image_avail_sem = device.create_semaphore()?;
        let render_finished_sem = device.create_semaphore()?;
        let in_flight_fen = device.create_fence()?;
        let uniforms = UniformBuffer::new(device)?;
        device.debug(|d| {
            d.set_object_name(device, &image_avail_sem, "Image available semaphore");
            d.set_object_name(device, &render_finished_sem, "Render finished semaphore");
            d.set_object_name(device, &in_flight_fen, "In-flight fence");
            d.set_object_name(device, &*uniforms.buffer, "Uniform buffer");
        });
        Ok(Self {
            image_avail_sem,
            render_finished_sem,
            in_flight_fen,
            uniforms,
            cmd_buffers: vec![],
        })
    }
}

impl Cleanup<VulkanDevice> for FrameState {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_semaphore(self.image_avail_sem, None);
        device.destroy_semaphore(self.render_finished_sem, None);
        device.destroy_fence(self.in_flight_fen, None);
        self.uniforms.cleanup(device);
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct UniformBufferObject {
    mvp: Mat4,
    viewproj_inv: Mat4,
}

pub struct Texture {
    image: VkImage,
    imgview: vk::ImageView,
    sampler: vk::Sampler,
}

impl Texture {
    pub fn new(
        device: &VulkanDevice, width: u32, height: u32, format: vk::Format, data: &[u8], sampler: vk::Sampler,
    ) -> VulkanResult<Self> {
        let image = device.create_image_from_data(width, height, format, ImageData::Single(data), Default::default())?;
        let imgview = device.create_image_view(*image, format, vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR)?;
        Ok(Self { image, imgview, sampler })
    }

    pub fn new_cubemap(
        device: &VulkanDevice, width: u32, height: u32, format: vk::Format, data: &[&[u8]; 6], sampler: vk::Sampler,
    ) -> VulkanResult<Self> {
        let image = device.create_image_from_data(width, height, format, ImageData::Array(data), vk::ImageCreateFlags::CUBE_COMPATIBLE)?;
        let imgview = device.create_image_view(*image, format, vk::ImageViewType::CUBE, vk::ImageAspectFlags::COLOR)?;
        Ok(Self { image, imgview, sampler })
    }

    pub fn update(&mut self, device: &VulkanDevice, x: u32, y: u32, width: u32, height: u32, data: &[u8]) -> VulkanResult<()> {
        //TODO: validate params
        device.update_image_from_data(*self.image, x as _, y as _, width, height, ImageData::Single(data))
    }

    pub(crate) fn descriptor(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image_view: self.imgview,
            sampler: self.sampler,
        }
    }
}

impl Cleanup<VulkanDevice> for Texture {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_image_view(self.imgview, None);
        self.image.cleanup(device);
    }
}

pub struct Shader {
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
}

impl Shader {
    pub fn new(device: &VulkanDevice, vert_spv: &[u32], frag_spv: &[u32]) -> VulkanResult<Self> {
        let vert = device.create_shader_module(vert_spv)?;
        let frag = device.create_shader_module(frag_spv)?;
        Ok(Self { vert, frag })
    }
}

impl Cleanup<VulkanDevice> for Shader {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_shader_module(self.vert, None);
        device.destroy_shader_module(self.frag, None);
    }
}
