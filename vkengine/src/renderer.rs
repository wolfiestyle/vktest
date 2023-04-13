use crate::create::CreateFromInfo;
use crate::device::{CubeData, ImageData, VkBuffer, VulkanDevice};
use crate::engine::{CmdBufferRing, DrawPayload, Shader, Texture, UploadBuffer, VulkanEngine};
use crate::pipeline::{Pipeline, PipelineMode};
use crate::types::{Cleanup, VulkanResult};
use crate::vertex::{IndexInput, VertexInput};
use ash::vk;
use bytemuck_derive::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Vec3, Vec4};
use inline_spirv::include_spirv;
use std::marker::PhantomData;
use std::mem::size_of;
use std::slice;
use std::sync::Arc;

#[derive(Debug)]
pub struct MeshRenderer<V, I> {
    device: Arc<VulkanDevice>,
    desc_layout: vk::DescriptorSetLayout,
    push_constants: vk::PushConstantRange,
    shader: Shader,
    pipeline: Pipeline,
    vertex_buffer: VkBuffer,
    index_buffer: VkBuffer,
    cmd_buffers: CmdBufferRing,
    obj_uniforms: UploadBuffer,
    material_buffer: VkBuffer,
    pub model: Affine3A,
    _p: PhantomData<(V, I)>,
}

impl<V: VertexInput, I: IndexInput> MeshRenderer<V, I> {
    pub fn new(
        engine: &VulkanEngine, vertices: &[V], indices: &[I], submeshes: &[MeshRenderSlice], transform: Affine3A,
    ) -> VulkanResult<Self> {
        let device = engine.device.clone();

        let shader = Shader::new(
            &device,
            include_spirv!("src/shaders/phong.vert.glsl", vert, glsl),
            include_spirv!("src/shaders/phong.frag.glsl", frag, glsl),
        )?;
        let desc_layout = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(3)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(4)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(5)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .build(),
            ])
            .create(&device)?;
        let push_constants = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: size_of::<u32>() as _,
        };
        let pipeline = Pipeline::builder(&shader)
            .vertex_input::<V>()
            .descriptor_layout(desc_layout)
            .push_constants(slice::from_ref(&push_constants))
            .render_to_swapchain(&engine.swapchain)
            .build(engine)?;

        let vertex_buffer = device.create_buffer_from_data(vertices, vk::BufferUsageFlags::VERTEX_BUFFER, "Vertex buffer")?;
        let index_buffer = device.create_buffer_from_data(indices, vk::BufferUsageFlags::INDEX_BUFFER, "Index buffer")?;

        let cmd_buffers = CmdBufferRing::new(&device)?;

        let obj_uniforms = UploadBuffer::new(
            &device,
            std::mem::size_of::<ObjectUniforms>() as _,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "MeshRenderer object uniforms",
        )?;

        let material_data: Vec<_> = submeshes
            .iter()
            .map(|subm| MaterialData {
                base_color: subm.base_color.into(),
                base_pbr: Vec4::new(1.0, subm.roughness, subm.metallic, 1.0),
                emiss_color: Vec3::from_array(subm.emissive).extend(subm.normal_scale),
            })
            .collect();
        let material_buffer =
            device.create_buffer_from_data(&material_data, vk::BufferUsageFlags::STORAGE_BUFFER, "MeshRenderer material data")?;

        Ok(Self {
            device,
            desc_layout,
            push_constants,
            shader,
            pipeline,
            vertex_buffer,
            index_buffer,
            cmd_buffers,
            obj_uniforms,
            material_buffer,
            model: transform,
            _p: PhantomData,
        })
    }

    pub fn render(&mut self, engine: &VulkanEngine, submeshes: &[MeshRenderSlice]) -> VulkanResult<DrawPayload> {
        let cmd_buffer = self.cmd_buffers.get_current_buffer(engine)?;
        engine.begin_secondary_draw_commands(cmd_buffer, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;

        let ubo = self.calc_uniforms(engine);
        self.obj_uniforms.map(engine)?.write_object(&ubo, 0);

        unsafe {
            // object
            self.device
                .debug(|d| d.cmd_begin_label(cmd_buffer, "3D object", [0.2, 0.6, 0.4, 1.0]));
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle);
            self.device.cmd_set_viewport(cmd_buffer, 0, &[engine.swapchain.viewport_inv()]);
            self.device.cmd_set_scissor(cmd_buffer, 0, &[engine.swapchain.extent_rect()]);
            let obj_buffer_info = self.obj_uniforms.descriptor(engine);
            let mat_buffer_info = self.material_buffer.descriptor();
            self.device.pushdesc_fn.cmd_push_descriptor_set(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &[
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(slice::from_ref(&obj_buffer_info))
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(slice::from_ref(&mat_buffer_info))
                        .build(),
                ],
            );
            self.device
                .cmd_bind_vertex_buffers(cmd_buffer, 0, slice::from_ref(&*self.vertex_buffer), &[0]);
            self.device
                .cmd_bind_index_buffer(cmd_buffer, self.index_buffer.handle, 0, I::VK_INDEX_TYPE);

            for (i, submesh) in submeshes.iter().enumerate() {
                let coltex_info = submesh.color_tex.unwrap_or_else(|| engine.default_texture.descriptor());
                let metrough_info = submesh.metal_rough_tex.unwrap_or_else(|| engine.default_texture.descriptor());
                let emiss_info = submesh.emiss_tex.unwrap_or_else(|| engine.default_texture.descriptor());
                let normal_info = submesh.normal_tex.unwrap_or_else(|| engine.default_normalmap.descriptor());
                self.device.pushdesc_fn.cmd_push_descriptor_set(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.layout,
                    0,
                    &[
                        vk::WriteDescriptorSet::builder()
                            .dst_binding(2)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(slice::from_ref(&coltex_info))
                            .build(),
                        vk::WriteDescriptorSet::builder()
                            .dst_binding(3)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(slice::from_ref(&metrough_info))
                            .build(),
                        vk::WriteDescriptorSet::builder()
                            .dst_binding(4)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(slice::from_ref(&emiss_info))
                            .build(),
                        vk::WriteDescriptorSet::builder()
                            .dst_binding(5)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(slice::from_ref(&normal_info))
                            .build(),
                    ],
                );
                let mat_id = i as u32;
                self.device.cmd_push_constants(
                    cmd_buffer,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&mat_id),
                );
                self.device
                    .cmd_draw_indexed(cmd_buffer, submesh.index_count, 1, submesh.index_offset, 0, 0);
            }

            self.device.debug(|d| d.cmd_end_label(cmd_buffer));
        }

        Ok(DrawPayload::new(engine.end_secondary_draw_commands(cmd_buffer)?))
    }

    fn calc_uniforms(&self, engine: &VulkanEngine) -> ObjectUniforms {
        ObjectUniforms {
            mvp: engine.view_proj * self.model,
            model: self.model.into(),
            light_dir: engine.sunlight.extend(0.1),
            light_color: Vec3::ONE.extend(0.1),
            view_pos: engine.camera.position.extend(0.1),
        }
    }

    pub fn rebuild_pipeline(&mut self, engine: &VulkanEngine) -> VulkanResult<()> {
        let pipeline = Pipeline::builder(&self.shader)
            .vertex_input::<V>()
            .descriptor_layout(self.desc_layout)
            .push_constants(slice::from_ref(&self.push_constants))
            .render_to_swapchain(&engine.swapchain)
            .build(engine)?;
        let old_pipeline = std::mem::replace(&mut self.pipeline, pipeline);
        self.device.dispose_of(old_pipeline);
        Ok(())
    }
}

impl<V, I> Drop for MeshRenderer<V, I> {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.vertex_buffer.cleanup(&self.device);
            self.index_buffer.cleanup(&self.device);
            self.pipeline.cleanup(&self.device);
            self.shader.cleanup(&self.device);
            self.desc_layout.cleanup(&self.device);
            self.cmd_buffers.cleanup(&self.device);
            self.obj_uniforms.cleanup(&self.device);
            self.material_buffer.cleanup(&self.device);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MeshRenderSlice {
    pub index_offset: u32,
    pub index_count: u32,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub normal_scale: f32,
    pub color_tex: Option<vk::DescriptorImageInfo>,
    pub metal_rough_tex: Option<vk::DescriptorImageInfo>,
    pub emiss_tex: Option<vk::DescriptorImageInfo>,
    pub normal_tex: Option<vk::DescriptorImageInfo>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
struct ObjectUniforms {
    mvp: Mat4,
    model: Mat4,
    light_dir: Vec4,
    light_color: Vec4,
    view_pos: Vec4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
struct MaterialData {
    base_color: Vec4,
    base_pbr: Vec4,
    emiss_color: Vec4,
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
    pub fn new(engine: &VulkanEngine, skybox_dims: (u32, u32), skybox_data: CubeData) -> VulkanResult<Self> {
        let device = engine.device.clone();
        let shader = Shader::new(
            &device,
            include_spirv!("src/shaders/skybox.vert.glsl", vert, glsl),
            include_spirv!("src/shaders/skybox.frag.glsl", frag, glsl),
        )?;
        let sampler = engine.get_sampler(
            vk::Filter::LINEAR,
            vk::Filter::LINEAR,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::REPEAT,
            false,
        )?;
        let desc_layout = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&[vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .immutable_samplers(slice::from_ref(&sampler))
                .build()])
            .create(&device)?;
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

        let texture = Texture::new(
            &device,
            skybox_dims.0,
            skybox_dims.1,
            vk::Format::R8G8B8A8_SRGB,
            ImageData::Cube(skybox_data),
            vk::Sampler::null(),
            false,
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
        let viewproj_inv = (engine.projection * engine.camera.get_view_rotation()).inverse();
        unsafe {
            // background
            self.device
                .debug(|d| d.cmd_begin_label(cmd_buffer, "Background", [0.2, 0.4, 0.6, 1.0]));
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle);
            self.device.cmd_set_viewport(cmd_buffer, 0, &[engine.swapchain.viewport_inv()]);
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
        let old_pipeline = std::mem::replace(&mut self.pipeline, pipeline);
        self.device.dispose_of(old_pipeline);
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
