use crate::camera::Camera;
use crate::device::{Swapchain, UniformBuffer, VkBuffer, VkImage, VulkanDevice};
use crate::types::*;
use ash::vk;
use cstr::cstr;
use glam::{Affine3A, Mat4};
use inline_spirv::include_spirv;
use std::marker::PhantomData;
use std::slice;
use std::time::{Duration, Instant};

const SWAPCHAIN_IMAGE_COUNT: u32 = 3;
const MAX_FRAMES_IN_FLIGHT: usize = 2;
//type Vertex = ([f32; 3], [f32; 3], [f32; 2]);
type Vertex = obj::TexturedVertex;

pub struct VulkanEngine {
    device: VulkanDevice,
    window_size: WinSize,
    window_resized: bool,
    swapchain: Swapchain,
    shader: Shader,
    pipeline: Pipeline<Vertex>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
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
        vk: VulkanDevice, window_size: WinSize, vertices: &[Vertex], indices: &[u32], img_width: u32, img_height: u32, img_data: &[u8],
    ) -> VulkanResult<Self> {
        let depth_format = vk.find_depth_format(false)?;
        let swapchain = vk.create_swapchain(window_size, SWAPCHAIN_IMAGE_COUNT, depth_format)?;
        eprintln!("color_format: {:?}, depth_format: {depth_format:?}", swapchain.format);

        let command_pool = vk.create_command_pool(vk.dev_info.graphics_idx, vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)?;
        let command_buffers = vk.create_command_buffers(command_pool, MAX_FRAMES_IN_FLIGHT as u32)?;
        let frame_state = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| FrameState::new(&vk))
            .collect::<Result<Vec<_>, _>>()?;

        let tex_sampler = vk.create_texture_sampler(vk::Filter::LINEAR, vk::SamplerAddressMode::REPEAT)?;
        let texture = Texture::new(&vk, img_width, img_height, img_data, tex_sampler)?;

        let shader = Shader::new(
            &vk,
            include_spirv!("src/shaders/texture.vert.glsl", vert, glsl),
            include_spirv!("src/shaders/texture.frag.glsl", frag, glsl),
            &[
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
            ],
        )?;
        let pipeline = Pipeline::new(&vk, &shader, &swapchain)?;

        let vertex_buffer = vk.create_buffer(vertices, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let index_buffer = vk.create_buffer(indices, vk::BufferUsageFlags::INDEX_BUFFER)?;

        let camera = Camera::default();
        let now = Instant::now();

        Ok(Self {
            device: vk,
            window_size,
            window_resized: false,
            swapchain,
            shader,
            pipeline,
            command_pool,
            command_buffers,
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

    fn record_command_buffer(&self, cmd_buffer: vk::CommandBuffer, image_idx: usize) -> VulkanResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .describe_err("Failed to begin recording command buffer")?;
        }

        // render pass
        let color_attach = vk::RenderingAttachmentInfo::builder()
            .image_view(self.swapchain.image_views[image_idx])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
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
            .render_area(self.swapchain.extent_rect())
            .layer_count(1)
            .color_attachments(slice::from_ref(&color_attach))
            .depth_attachment(&depth_attach);

        // descriptor set
        let uniforms = &self.frame_state[self.current_frame].uniforms;
        let buffer_info = vk::DescriptorBufferInfo {
            buffer: *uniforms.buffer,
            offset: 0,
            range: uniforms.buffer_size() as _,
        };
        let image_info = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image_view: self.texture.imgview,
            sampler: self.texture.sampler,
        };
        let desc_writes = [
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
        ];

        // commands
        unsafe {
            self.device.transition_image_layout(
                cmd_buffer,
                self.swapchain.images[image_idx],
                self.swapchain.format,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
            self.device
                .debug(|d| d.cmd_begin_label(cmd_buffer, "3D object", [0.2, 0.4, 0.6, 1.0]));
            self.device.dynrender_fn.cmd_begin_rendering(cmd_buffer, &render_info);
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle);
            self.device
                .cmd_bind_vertex_buffers(cmd_buffer, 0, slice::from_ref(&*self.vertex_buffer), &[0]);
            self.device
                .cmd_bind_index_buffer(cmd_buffer, *self.index_buffer, 0, vk::IndexType::UINT32);
            self.device.pushdesc_fn.cmd_push_descriptor_set(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &desc_writes,
            );
            self.device
                .cmd_set_viewport(cmd_buffer, 0, slice::from_ref(&self.swapchain.viewport()));
            self.device
                .cmd_set_scissor(cmd_buffer, 0, slice::from_ref(&self.swapchain.extent_rect()));
            self.device.cmd_draw_indexed(cmd_buffer, self.index_count, 1, 0, 0, 0);
            self.device.dynrender_fn.cmd_end_rendering(cmd_buffer);
            self.device.debug(|d| d.cmd_end_label(cmd_buffer));
            self.device.transition_image_layout(
                cmd_buffer,
                self.swapchain.images[image_idx],
                self.swapchain.format,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
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
        let command_buffer = self.command_buffers[self.current_frame];

        let image_idx = unsafe {
            self.device
                .wait_for_fences(slice::from_ref(&in_flight_fen), true, u64::MAX)
                .describe_err("Failed waiting for fence")?;
            self.prev_frame_time = self.last_frame_time;
            self.last_frame_time = Instant::now();
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

        self.update_uniforms();

        unsafe {
            self.device
                .reset_fences(slice::from_ref(&in_flight_fen))
                .describe_err("Failed resetting fences")?;
            self.device
                .reset_command_buffer(command_buffer, Default::default())
                .describe_err("Failed to reset command buffer")?;
        }

        self.record_command_buffer(command_buffer, image_idx as _)?;

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

    fn update_uniforms(&mut self) {
        let view = self.camera.get_view_transform();
        let proj = Mat4::perspective_rh(45.0f32.to_radians(), self.swapchain.aspect(), 0.1, 1000.0);
        let ubo = UniformBufferObject {
            mvp: proj * view * self.model,
        };
        self.frame_state[self.current_frame].uniforms.write_uniforms(ubo);
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
            self.frame_state.cleanup(&self.device);
            self.device.destroy_command_pool(self.command_pool, None);
            self.pipeline.cleanup(&self.device);
            self.shader.cleanup(&self.device);
            self.swapchain.cleanup(&self.device);
        }
    }
}

struct Pipeline<Vert> {
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    _p: PhantomData<Vert>,
}

impl<Vert: VertexAttrDesc> Pipeline<Vert> {
    fn new(device: &VulkanDevice, shader: &Shader, swapchain: &Swapchain) -> VulkanResult<Self> {
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

        let binding_desc = Vert::binding_desc(0);
        let attr_desc = Vert::attr_desc(0);
        let vertex_input_ci = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(slice::from_ref(&binding_desc))
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
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blend_ci = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(slice::from_ref(&color_attach));

        let depth_stencil_ci = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);

        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::builder().set_layouts(slice::from_ref(&shader.desc_layout));

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_ci, None)
                .describe_err("Failed to create pipeline layout")?
        };

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
            .layout(pipeline_layout)
            .push_next(&mut pipeline_rendering_ci);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), slice::from_ref(&pipeline_ci), None)
                .map_err(|(_, err)| VkError::VulkanMsg("Error creating pipeline", err))?
        };

        device.debug(|d| d.set_object_name(&device, &pipeline[0], &format!("Pipeline<{}>", std::any::type_name::<Vert>())));

        Ok(Self {
            handle: pipeline[0],
            layout: pipeline_layout,
            _p: PhantomData,
        })
    }
}

impl<V> Cleanup<VulkanDevice> for Pipeline<V> {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_pipeline(self.handle, None);
        device.destroy_pipeline_layout(self.layout, None);
    }
}

struct FrameState {
    image_avail_sem: vk::Semaphore,
    render_finished_sem: vk::Semaphore,
    in_flight_fen: vk::Fence,
    uniforms: UniformBuffer<UniformBufferObject>,
}

impl FrameState {
    fn new(device: &VulkanDevice) -> VulkanResult<Self> {
        let image_avail_sem = device.create_semaphore()?;
        let render_finished_sem = device.create_semaphore()?;
        let in_flight_fen = device.create_fence()?;
        let uniforms = device.create_uniform_buffer()?;
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
}

struct Texture {
    image: VkImage,
    imgview: vk::ImageView,
    sampler: vk::Sampler,
}

impl Texture {
    fn new(device: &VulkanDevice, width: u32, height: u32, data: &[u8], sampler: vk::Sampler) -> VulkanResult<Self> {
        let image = device.create_image_from_data(width, height, data)?;
        let imgview = device.create_image_view(*image, vk::Format::R8G8B8A8_SRGB, vk::ImageAspectFlags::COLOR)?;
        Ok(Self { image, imgview, sampler })
    }
}

impl Cleanup<VulkanDevice> for Texture {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_image_view(self.imgview, None);
        self.image.cleanup(device);
    }
}

struct Shader {
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
    desc_layout: vk::DescriptorSetLayout,
}

impl Shader {
    fn new(device: &VulkanDevice, vert_spv: &[u32], frag_spv: &[u32], bindings: &[vk::DescriptorSetLayoutBinding]) -> VulkanResult<Self> {
        let vert = device.create_shader_module(vert_spv)?;
        let frag = device.create_shader_module(frag_spv)?;
        let desc_layout = device.create_descriptor_set_layout(bindings)?;
        Ok(Self { vert, frag, desc_layout })
    }
}

impl Cleanup<VulkanDevice> for Shader {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_shader_module(self.vert, None);
        device.destroy_shader_module(self.frag, None);
        device.destroy_descriptor_set_layout(self.desc_layout, None);
    }
}
