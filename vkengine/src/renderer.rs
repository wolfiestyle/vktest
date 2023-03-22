use crate::device::{VkBuffer, VulkanDevice};
use crate::engine::{CmdBufferRing, DrawPayload, Pipeline, PipelineMode, Shader, Texture, UploadBuffer, VulkanEngine};
use crate::types::{Cleanup, VulkanResult};
use crate::vertex::{IndexInput, VertexInput};
use ash::vk;
use bytemuck_derive::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Vec3, Vec4};
use inline_spirv::include_spirv;
use std::marker::PhantomData;
use std::slice;
use std::sync::Arc;

pub struct MeshRenderer<V, I> {
    device: Arc<VulkanDevice>,
    desc_layout: vk::DescriptorSetLayout,
    shader: Shader,
    pipeline: Pipeline,
    vertex_buffer: VkBuffer,
    index_buffer: Option<VkBuffer>,
    elem_count: u32,
    base_color: [f32; 4],
    texture: Option<Texture>,
    cmd_buffers: CmdBufferRing,
    uniforms: UploadBuffer,
    pub model: Affine3A,
    _p: PhantomData<(V, I)>,
}

impl<V: VertexInput, I: IndexInput> MeshRenderer<V, I> {
    pub fn new(
        engine: &VulkanEngine, vertices: &[V], indices: Option<&[I]>, base_color: [f32; 4], texture: Option<Texture>,
    ) -> VulkanResult<Self> {
        let device = engine.device.clone();
        let shader = Shader::new(
            &device,
            include_spirv!("src/shaders/texture.vert.glsl", vert, glsl),
            include_spirv!("src/shaders/texture.frag.glsl", frag, glsl),
        )?;
        let desc_layout = device.create_descriptor_set_layout(&[
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
        ])?;
        let pipeline = Pipeline::builder(&shader)
            .vertex_input::<V>()
            .descriptor_layout(desc_layout)
            .render_to_swapchain(&engine.swapchain)
            .build(engine)?;

        let vertex_buffer = device.create_buffer_from_data(vertices, vk::BufferUsageFlags::VERTEX_BUFFER, "Vertex buffer")?;
        let index_buffer = indices
            .map(|idx| device.create_buffer_from_data(idx, vk::BufferUsageFlags::INDEX_BUFFER, "Index buffer"))
            .transpose()?;
        let elem_count = indices.map(|idx| idx.len() as u32).unwrap_or_else(|| vertices.len() as u32);

        let cmd_buffers = CmdBufferRing::new(&device)?;

        let uniforms = UploadBuffer::new(
            &device,
            std::mem::size_of::<ObjectUniforms>() as _,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "MeshRenderer uniform buffer",
        )?;

        Ok(Self {
            device,
            desc_layout,
            shader,
            pipeline,
            vertex_buffer,
            index_buffer,
            elem_count,
            base_color,
            texture,
            cmd_buffers,
            uniforms,
            model: Affine3A::IDENTITY,
            _p: PhantomData,
        })
    }

    pub fn render(&mut self, engine: &VulkanEngine) -> VulkanResult<DrawPayload> {
        let cmd_buffer = self.cmd_buffers.get_current_buffer(engine)?;
        engine.begin_secondary_draw_commands(cmd_buffer, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;

        let ubo = self.calc_uniforms(engine);
        self.uniforms.map(engine)?.write_object(&ubo);

        let buffer_info = self.uniforms.descriptor(engine);
        let image_info = self.texture.as_ref().unwrap_or(&engine.default_texture).descriptor();
        unsafe {
            // object
            self.device
                .debug(|d| d.cmd_begin_label(cmd_buffer, "3D object", [0.2, 0.6, 0.4, 1.0]));
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle);
            self.device.cmd_set_viewport(cmd_buffer, 0, &[engine.swapchain.viewport()]);
            self.device.cmd_set_scissor(cmd_buffer, 0, &[engine.swapchain.extent_rect()]);
            self.device.pushdesc_fn.cmd_push_descriptor_set(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
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
            if let Some(index_buffer) = &self.index_buffer {
                self.device
                    .cmd_bind_index_buffer(cmd_buffer, index_buffer.handle, 0, I::VK_INDEX_TYPE);
                self.device.cmd_draw_indexed(cmd_buffer, self.elem_count, 1, 0, 0, 0);
            } else {
                self.device.cmd_draw(cmd_buffer, self.elem_count, 1, 0, 0);
            }
            self.device.debug(|d| d.cmd_end_label(cmd_buffer));
        }

        Ok(DrawPayload::new(engine.end_secondary_draw_commands(cmd_buffer)?))
    }

    fn calc_uniforms(&self, engine: &VulkanEngine) -> ObjectUniforms {
        ObjectUniforms {
            mvp: engine.view_proj * self.model,
            light_dir: engine.sunlight.extend(self.base_color[0]),
            light_color: Vec3::ONE.extend(self.base_color[1]),
            ambient: Vec3::splat(0.1).extend(self.base_color[2]),
        }
    }

    pub fn rebuild_pipeline(&mut self, engine: &VulkanEngine) -> VulkanResult<()> {
        let pipeline = Pipeline::builder(&self.shader)
            .vertex_input::<V>()
            .descriptor_layout(self.desc_layout)
            .render_to_swapchain(&engine.swapchain)
            .build(engine)?;
        unsafe {
            self.pipeline.cleanup(&self.device);
        }
        self.pipeline = pipeline;
        Ok(())
    }
}

impl<V, I> Drop for MeshRenderer<V, I> {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.vertex_buffer.cleanup(&self.device);
            self.index_buffer.cleanup(&self.device);
            self.texture.cleanup(&self.device);
            self.pipeline.cleanup(&self.device);
            self.shader.cleanup(&self.device);
            self.desc_layout.cleanup(&self.device);
            self.cmd_buffers.cleanup(&self.device);
            self.uniforms.cleanup(&self.device);
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
struct ObjectUniforms {
    mvp: Mat4,
    light_dir: Vec4,
    light_color: Vec4,
    ambient: Vec4,
}

pub struct SkyboxRenderer {
    device: Arc<VulkanDevice>,
    desc_layout: vk::DescriptorSetLayout,
    push_constants: vk::PushConstantRange,
    shader: Shader,
    pipeline: Pipeline,
    texture: Texture,
    cmd_buffers: CmdBufferRing,
}

impl SkyboxRenderer {
    pub fn new(engine: &VulkanEngine, skybox_dims: (u32, u32), skybox_data: &[&[u8]; 6]) -> VulkanResult<Self> {
        let device = engine.device.clone();
        let shader = Shader::new(
            &device,
            include_spirv!("src/shaders/skybox.vert.glsl", vert, glsl),
            include_spirv!("src/shaders/skybox.frag.glsl", frag, glsl),
        )?;
        let sampler = engine.get_sampler(vk::Filter::LINEAR, vk::Filter::LINEAR, vk::SamplerAddressMode::REPEAT, false)?;
        let desc_layout = device.create_descriptor_set_layout(&[vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .immutable_samplers(slice::from_ref(&sampler))
            .build()])?;
        let push_constants = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<Mat4>() as _)
            .build();
        let pipeline = Pipeline::builder(&shader)
            .descriptor_layout(desc_layout)
            .push_constants(slice::from_ref(&push_constants))
            .render_to_swapchain(&engine.swapchain)
            .mode(PipelineMode::Background)
            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
            .build(engine)?;

        let texture = Texture::new_cubemap(
            &device,
            skybox_dims.0,
            skybox_dims.1,
            vk::Format::R8G8B8A8_SRGB,
            skybox_data,
            vk::Sampler::null(),
        )?;

        let cmd_buffers = CmdBufferRing::new(&device)?;

        Ok(Self {
            device,
            desc_layout,
            push_constants,
            shader,
            pipeline,
            texture,
            cmd_buffers,
        })
    }

    pub fn render(&mut self, engine: &VulkanEngine) -> VulkanResult<DrawPayload> {
        let cmd_buffer = self.cmd_buffers.get_current_buffer(engine)?;
        engine.begin_secondary_draw_commands(cmd_buffer, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;

        let image_info = self.texture.descriptor();
        let viewproj_inv = engine.view_proj.inverse();
        unsafe {
            // background
            self.device
                .debug(|d| d.cmd_begin_label(cmd_buffer, "Background", [0.2, 0.4, 0.6, 1.0]));
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle);
            self.device.cmd_set_viewport(cmd_buffer, 0, &[engine.swapchain.viewport()]);
            self.device.cmd_set_scissor(cmd_buffer, 0, &[engine.swapchain.extent_rect()]);
            self.device.pushdesc_fn.cmd_push_descriptor_set(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &[vk::WriteDescriptorSet::builder()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(slice::from_ref(&image_info))
                    .build()],
            );
            self.device.cmd_push_constants(
                cmd_buffer,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&viewproj_inv),
            );
            self.device.cmd_draw(cmd_buffer, 4, 1, 0, 0);
            self.device.debug(|d| d.cmd_end_label(cmd_buffer));
        }

        Ok(DrawPayload::new(engine.end_secondary_draw_commands(cmd_buffer)?))
    }

    pub fn rebuild_pipeline(&mut self, engine: &VulkanEngine) -> VulkanResult<()> {
        let pipeline = Pipeline::builder(&self.shader)
            .descriptor_layout(self.desc_layout)
            .push_constants(slice::from_ref(&self.push_constants))
            .render_to_swapchain(&engine.swapchain)
            .mode(PipelineMode::Background)
            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
            .build(engine)?;
        unsafe {
            self.pipeline.cleanup(&self.device);
        }
        self.pipeline = pipeline;
        Ok(())
    }
}

impl Drop for SkyboxRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.texture.cleanup(&self.device);
            self.pipeline.cleanup(&self.device);
            self.shader.cleanup(&self.device);
            self.desc_layout.cleanup(&self.device);
            self.cmd_buffers.cleanup(&self.device);
        }
    }
}
