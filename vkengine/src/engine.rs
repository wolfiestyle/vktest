use crate::device::{Swapchain, UniformBuffer, VkBuffer, VkImage, VulkanDevice};
use crate::types::*;
use ash::vk;
use cstr::cstr;
use glam::{Affine3A, Mat4, Vec3};
use inline_spirv::include_spirv;
use std::slice;
use std::time::Instant;

const SWAPCHAIN_IMAGE_COUNT: u32 = 3;
const MAX_FRAMES_IN_FLIGHT: usize = 2;
const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0;
//type Vertex = ([f32; 3], [f32; 3], [f32; 2]);
type Vertex = obj::TexturedVertex;

pub struct VulkanEngine {
    device: VulkanDevice,
    window_size: WinSize,
    window_resized: bool,
    swapchain: Swapchain,
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
    view: Affine3A,
}

impl VulkanEngine {
    pub fn new(
        vk: VulkanDevice, window_size: WinSize, vertices: &[Vertex], indices: &[u32], img_width: u32, img_height: u32, img_data: &[u8],
    ) -> VulkanResult<Self> {
        let vert_spv = include_spirv!("src/shaders/texture.vert.glsl", vert, glsl);
        let frag_spv = include_spirv!("src/shaders/texture.frag.glsl", frag, glsl);

        let depth_format = vk.find_depth_format(false)?;
        let swapchain = vk.create_swapchain(window_size, SWAPCHAIN_IMAGE_COUNT, depth_format)?;
        eprintln!("color_format: {:?}, depth_format: {depth_format:?}", swapchain.format);

        let command_buffers = vk.create_command_buffers(MAX_FRAMES_IN_FLIGHT as u32)?;
        let frame_state = command_buffers
            .into_iter()
            .map(|cmd_buf| FrameState::new(&vk, cmd_buf))
            .collect::<Result<Vec<_>, _>>()?;

        let texture = vk.create_image_from_data(img_width, img_height, img_data)?;
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
            &swapchain,
            pipeline_layout,
        )?;

        let vertex_buffer = vk.create_buffer(vertices, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let index_buffer = vk.create_buffer(indices, vk::BufferUsageFlags::INDEX_BUFFER)?;

        let view = Affine3A::look_at_rh(Vec3::splat(2.0), Vec3::ZERO, Vec3::NEG_Z);

        let now = Instant::now();

        Ok(Self {
            device: vk,
            window_size,
            window_resized: false,
            swapchain,
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
            view,
        })
    }

    pub fn resize(&mut self, window_size: WinSize) {
        eprintln!("window size: {} x {}", window_size.width, window_size.height);
        if window_size != self.window_size {
            self.window_size = window_size;
            self.window_resized = true;
        }
    }

    pub fn get_frame_time(&self) -> Instant {
        self.frame_time
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
        let alloc_info = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(pool).set_layouts(layouts);
        let desc_sets = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .describe_err("Failed to allocate descriptor sets")?
        };

        for (&desc_set, fstate) in desc_sets.iter().zip(frame_state) {
            let buffer_info = vk::DescriptorBufferInfo {
                buffer: *fstate.uniforms.buffer,
                offset: 0,
                range: fstate.uniforms.buffer_size() as _,
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
                    .buffer_info(slice::from_ref(&buffer_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(desc_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(slice::from_ref(&image_info))
                    .build(),
            ];
            unsafe {
                device.update_descriptor_sets(&descr_write, &[]);
            }
        }

        Ok(desc_sets)
    }

    fn create_pipeline_layout(device: &VulkanDevice, desc_set_layout: vk::DescriptorSetLayout) -> VulkanResult<vk::PipelineLayout> {
        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::builder().set_layouts(slice::from_ref(&desc_set_layout));

        unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_ci, None)
                .describe_err("Failed to create pipeline layout")
        }
    }

    fn create_graphics_pipeline(
        device: &VulkanDevice, vert_shader_spv: &[u32], frag_shader_spv: &[u32], binding_desc: &[vk::VertexInputBindingDescription],
        attr_desc: &[vk::VertexInputAttributeDescription], swapchain: &Swapchain, pipeline_layout: vk::PipelineLayout,
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

        unsafe {
            device.destroy_shader_module(vert_shader, None);
            device.destroy_shader_module(frag_shader, None);
        }

        Ok(pipeline[0])
    }

    fn record_command_buffer(&self, cmd_buffer: vk::CommandBuffer, image_idx: usize) -> VulkanResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .describe_err("Failed to begin recording command buffer")?;
        }

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

        unsafe {
            self.device
                .debug(|d| d.cmd_begin_label(cmd_buffer, "3D object", [0.2, 0.4, 0.6, 1.0]));
            self.device.transition_image_layout(
                cmd_buffer,
                self.swapchain.images[image_idx],
                self.swapchain.format,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
            self.device.dynrender_fn.cmd_begin_rendering(cmd_buffer, &render_info);
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device
                .cmd_bind_vertex_buffers(cmd_buffer, 0, slice::from_ref(&*self.vertex_buffer), &[0]);
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
                .cmd_set_viewport(cmd_buffer, 0, slice::from_ref(&self.swapchain.viewport()));
            self.device
                .cmd_set_scissor(cmd_buffer, 0, slice::from_ref(&self.swapchain.extent_rect()));
            self.device.cmd_draw_indexed(cmd_buffer, self.index_count, 1, 0, 0, 0);
            self.device.dynrender_fn.cmd_end_rendering(cmd_buffer);
            self.device.transition_image_layout(
                cmd_buffer,
                self.swapchain.images[image_idx],
                self.swapchain.format,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
            self.device.debug(|d| d.cmd_end_label(cmd_buffer));
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
                .wait_for_fences(slice::from_ref(&in_flight_fen), true, u64::MAX)
                .describe_err("Failed waiting for fence")?;
            self.frame_time = Instant::now();
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
        let time = ((self.frame_time - self.start_time).as_micros() as f64 / 1000000.0) as f32;
        let model = Affine3A::from_axis_angle(Vec3::Z, 90.0 * DEG_TO_RAD * time);
        let proj = Mat4::perspective_rh(45.0 * DEG_TO_RAD, self.swapchain.aspect(), 0.1, 1000.0);
        let ubo = UniformBufferObject {
            mvp: proj * self.view * model,
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
            self.device.destroy_sampler(self.tex_sampler, None);
            self.device.destroy_image_view(self.tex_imgview, None);
            self.texture.cleanup(&self.device);
            self.frame_state.cleanup(&self.device);
            self.device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.swapchain.cleanup(&self.device);
        }
    }
}

struct FrameState {
    image_avail_sem: vk::Semaphore,
    render_finished_sem: vk::Semaphore,
    in_flight_fen: vk::Fence,
    uniforms: UniformBuffer<UniformBufferObject>,
    command_buffer: vk::CommandBuffer,
}

impl FrameState {
    fn new(device: &VulkanDevice, command_buffer: vk::CommandBuffer) -> VulkanResult<Self> {
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
            command_buffer,
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
