use crate::create::CreateFromInfo;
use crate::device::{VkBuffer, VulkanDevice};
use crate::engine::{CmdBufferRing, DrawPayload, SamplerOptions, UploadBuffer, VulkanEngine};
use crate::pipeline::{Pipeline, PipelineMode, Shader};
use crate::texture::Texture;
use crate::types::{Cleanup, VulkanResult};
use crate::vertex::{IndexInput, VertexInput};
use ash::vk;
use bytemuck_derive::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Vec3, Vec4};
use gltf_import::LightType;
use inline_spirv::include_spirv;
use std::marker::PhantomData;
use std::mem::size_of;
use std::slice;
use std::sync::Arc;

#[derive(Debug)]
pub struct MeshRenderer<V, I> {
    device: Arc<VulkanDevice>,
    push_desc_layout: vk::DescriptorSetLayout,
    push_constants: vk::PushConstantRange,
    shader: Shader,
    pipeline: Pipeline,
    vertex_buffer: VkBuffer,
    index_buffer: VkBuffer,
    cmd_buffers: CmdBufferRing,
    obj_uniforms: UploadBuffer,
    pub model: Affine3A,
    _p: PhantomData<(V, I)>,
}

impl<V: VertexInput, I: IndexInput> MeshRenderer<V, I> {
    pub fn new(engine: &VulkanEngine, vertices: &[V], indices: &[I], transform: Affine3A) -> VulkanResult<Self> {
        let device = engine.device.clone();

        let shader = Shader::new(
            &device,
            include_spirv!("src/shaders/pbr.vert.glsl", vert, glsl),
            include_spirv!("src/shaders/pbr.frag.glsl", frag, glsl),
        )?;
        let sampler = engine.get_sampler(SamplerOptions {
            wrap_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            wrap_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            ..Default::default()
        })?;
        let push_desc_layout = vk::DescriptorSetLayoutCreateInfo::builder()
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
                    .immutable_samplers(slice::from_ref(&sampler))
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(3)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .immutable_samplers(slice::from_ref(&sampler))
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(4)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .immutable_samplers(slice::from_ref(&sampler))
                    .build(),
            ])
            .create(&device)?;
        let push_constants = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: size_of::<MaterialData>() as _,
        };
        let pipeline = Pipeline::builder_graphics(&shader)
            .vertex_input::<V>()
            .descriptor_layouts(&[push_desc_layout, engine.image_desc_layout])
            .push_constants(slice::from_ref(&push_constants))
            .render_to_swapchain(&engine.swapchain)
            .build(engine)?;

        let vertex_buffer = device.create_buffer_from_data(vertices, vk::BufferUsageFlags::VERTEX_BUFFER, "Vertex buffer")?;
        let index_buffer = device.create_buffer_from_data(indices, vk::BufferUsageFlags::INDEX_BUFFER, "Index buffer")?;

        let cmd_buffers = CmdBufferRing::new(&device)?;

        let obj_uniforms = UploadBuffer::new(
            &device,
            size_of::<ObjectUniforms>() as _,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "MeshRenderer object uniforms",
        )?;

        Ok(Self {
            device,
            push_desc_layout,
            push_constants,
            shader,
            pipeline,
            vertex_buffer,
            index_buffer,
            cmd_buffers,
            obj_uniforms,
            model: transform,
            _p: PhantomData,
        })
    }

    pub fn render(
        &mut self, engine: &VulkanEngine, submeshes: &[MeshRenderData], irrmap: &Texture, prefmap: &Texture, brdf_lut: &Texture,
    ) -> VulkanResult<DrawPayload> {
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
            let light_buffer_info = engine.light_buffer.descriptor(engine);
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
                        .buffer_info(slice::from_ref(&light_buffer_info))
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(slice::from_ref(&irrmap.info))
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(3)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(slice::from_ref(&prefmap.info))
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(4)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(slice::from_ref(&brdf_lut.info))
                        .build(),
                ],
            );
            self.device
                .cmd_bind_vertex_buffers(cmd_buffer, 0, slice::from_ref(&*self.vertex_buffer), &[0]);
            self.device
                .cmd_bind_index_buffer(cmd_buffer, self.index_buffer.handle, 0, I::VK_INDEX_TYPE);

            for submesh in submeshes {
                self.device.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.layout,
                    1,
                    slice::from_ref(&submesh.descriptor),
                    &[],
                );
                self.device.cmd_push_constants(
                    cmd_buffer,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&submesh.material_data),
                );
                self.device.cmd_draw_indexed(
                    cmd_buffer,
                    submesh.index_count,
                    1,
                    submesh.index_offset,
                    submesh.vertex_offset as _,
                    0,
                );
            }

            self.device.debug(|d| d.cmd_end_label(cmd_buffer));
        }

        Ok(DrawPayload::new(engine.end_secondary_draw_commands(cmd_buffer)?))
    }

    fn calc_uniforms(&self, engine: &VulkanEngine) -> ObjectUniforms {
        ObjectUniforms {
            mvp: engine.view_proj * self.model,
            model: self.model.into(),
            view_pos: engine.camera.position,
            num_lights: engine.lights.len() as _,
        }
    }

    pub fn rebuild_pipeline(&mut self, engine: &VulkanEngine) -> VulkanResult<()> {
        let pipeline = Pipeline::builder_graphics(&self.shader)
            .vertex_input::<V>()
            .descriptor_layouts(&[self.push_desc_layout, engine.image_desc_layout])
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
            self.push_desc_layout.cleanup(&self.device);
            self.cmd_buffers.cleanup(&self.device);
            self.obj_uniforms.cleanup(&self.device);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MeshRenderData {
    pub index_offset: u32,
    pub index_count: u32,
    pub vertex_offset: u32,
    pub descriptor: vk::DescriptorSet,
    pub material_data: MaterialData,
}

impl MeshRenderData {
    pub fn from_gltf(submesh: &gltf_import::Submesh, material: &gltf_import::Material, descriptor: vk::DescriptorSet) -> Self {
        Self {
            index_offset: submesh.index_offset,
            index_count: submesh.index_count,
            vertex_offset: submesh.vertex_offset,
            descriptor,
            material_data: MaterialData::from_gltf(material),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LightData {
    pub pos: Vec4, // .w: 0 = directional, 1 = point/spot
    pub color: Vec3,
    pub spot_offset: f32,
    pub spot_dir: Vec3,
    pub spot_scale: f32,
}

impl LightData {
    pub fn point(pos: Vec3, color: Vec3) -> Self {
        Self {
            pos: pos.extend(1.0),
            color,
            spot_offset: 1.0,
            spot_dir: Vec3::ZERO,
            spot_scale: 0.0,
        }
    }

    pub fn directional(dir: Vec3, color: Vec3) -> Self {
        Self {
            pos: dir.extend(0.0),
            color,
            spot_offset: 1.0,
            spot_dir: Vec3::ZERO,
            spot_scale: 0.0,
        }
    }

    pub fn spot(pos: Vec3, dir: Vec3, color: Vec3, inner_angle: f32, outer_angle: f32) -> Self {
        let [spot_scale, spot_offset] = Self::calc_spot(inner_angle, outer_angle);
        Self {
            pos: pos.extend(1.0),
            color,
            spot_offset,
            spot_dir: dir,
            spot_scale,
        }
    }

    fn calc_spot(inner_angle: f32, outer_angle: f32) -> [f32; 2] {
        let spot_scale = 1.0 / f32::max(0.001, inner_angle.cos() - outer_angle.cos());
        let spot_offset = -outer_angle.cos() * spot_scale;
        [spot_scale, spot_offset]
    }

    pub fn from_gltf(light: &gltf_import::Light, node: &gltf_import::Node) -> Self {
        let color = Vec3::from(light.color) * light.intensity;
        match light.type_ {
            LightType::Directional => LightData::directional(node.transform.transform_vector3(Vec3::NEG_Z).normalize_or_zero(), color),
            LightType::Point => LightData::point(node.transform.translation.into(), color),
            LightType::Spot { inner_angle, outer_angle } => LightData::spot(
                node.transform.translation.into(),
                node.transform.transform_vector3(Vec3::NEG_Z).normalize_or_zero(),
                color,
                inner_angle,
                outer_angle,
            ),
        }
    }

    pub fn is_directional(&self) -> bool {
        self.pos.w == 0.0
    }

    pub fn is_point(&self) -> bool {
        self.pos.w != 0.0 && self.spot_scale == 0.0
    }

    pub fn is_spot(&self) -> bool {
        self.spot_scale != 0.0
    }

    pub fn set_type(&mut self, type_: gltf_import::LightType) {
        match type_ {
            LightType::Directional => {
                self.pos.w = 0.0;
                self.spot_scale = 0.0;
                self.spot_offset = 1.0;
            }
            LightType::Point => {
                self.pos.w = 1.0;
                self.spot_scale = 0.0;
                self.spot_offset = 1.0;
            }
            LightType::Spot { inner_angle, outer_angle } => {
                let [scale, offset] = Self::calc_spot(inner_angle, outer_angle);
                self.pos.w = 1.0;
                self.spot_scale = scale;
                self.spot_offset = offset;
                if self.spot_dir == Vec3::ZERO {
                    self.spot_dir = Vec3::NEG_Y;
                }
            }
        }
    }
}

impl Default for LightData {
    fn default() -> Self {
        Self {
            pos: Vec4::W,
            color: Vec3::ONE,
            spot_offset: 1.0,
            spot_dir: Vec3::ZERO,
            spot_scale: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
struct ObjectUniforms {
    mvp: Mat4,
    model: Mat4,
    view_pos: Vec3,
    num_lights: u32,
}

const UV_BITS: u32 = 1;
const UV_MASK: u32 = (1 << UV_BITS) - 1;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct MaterialData {
    pub base_color: Vec4,
    pub base_pbr: Vec3,
    pub normal_scale: f32,
    pub emissive: Vec3,
    pub uv_sets: u32,
}

impl MaterialData {
    pub fn from_gltf(material: &gltf_import::Material) -> Self {
        let color_uv = material.color_tex.map(|tex| tex.uv_set & UV_MASK).unwrap_or_default();
        let metrough_uv = material.metallic_roughness_tex.map(|tex| tex.uv_set & UV_MASK).unwrap_or_default();
        let normal_uv = material.normal_tex.map(|tex| tex.uv_set & UV_MASK).unwrap_or_default();
        let emiss_uv = material.emissive_tex.map(|tex| tex.uv_set & UV_MASK).unwrap_or_default();
        let occl_uv = material.occlusion_tex.map(|tex| tex.uv_set & UV_MASK).unwrap_or_default();
        Self {
            base_color: material.base_color.into(),
            base_pbr: Vec3::new(material.occlusion_strength, material.roughness, material.metallic),
            normal_scale: material.normal_scale,
            emissive: material.emissive.into(),
            uv_sets: (color_uv)
                | (metrough_uv << UV_BITS)
                | (normal_uv << UV_BITS * 2)
                | (emiss_uv << UV_BITS * 3)
                | (occl_uv << UV_BITS * 4),
        }
    }
}

pub struct SkyboxRenderer {
    device: Arc<VulkanDevice>,
    desc_layout: vk::DescriptorSetLayout,
    push_constants: vk::PushConstantRange,
    shader: Shader,
    pipeline: Pipeline,
    cmd_buffers: CmdBufferRing,
    pub lod: f32,
}

impl SkyboxRenderer {
    pub fn new(engine: &VulkanEngine) -> VulkanResult<Self> {
        let device = engine.device.clone();
        let shader = Shader::new(
            &device,
            include_spirv!("src/shaders/skybox.vert.glsl", vert, glsl),
            include_spirv!("src/shaders/skybox.frag.glsl", frag, glsl),
        )?;
        let sampler = engine.get_sampler(SamplerOptions {
            wrap_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            wrap_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            ..Default::default()
        })?;
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
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<SkyboxParams>() as _)
            .build();
        let pipeline = Pipeline::builder_graphics(&shader)
            .descriptor_layout(&desc_layout)
            .push_constants(slice::from_ref(&push_constants))
            .render_to_swapchain(&engine.swapchain)
            .mode(PipelineMode::Background)
            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
            .build(engine)?;

        let cmd_buffers = CmdBufferRing::new(&device)?;

        Ok(Self {
            device,
            desc_layout,
            push_constants,
            shader,
            pipeline,
            cmd_buffers,
            lod: 0.0,
        })
    }

    pub fn render(&mut self, engine: &VulkanEngine, cubemap: &Texture) -> VulkanResult<DrawPayload> {
        let cmd_buffer = self.cmd_buffers.get_current_buffer(engine)?;
        engine.begin_secondary_draw_commands(cmd_buffer, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;

        let params = SkyboxParams {
            viewproj_inv: (engine.projection * engine.camera.get_view_rotation()).inverse(),
            lod: Vec4::new(self.lod, 0.0, 0.0, 0.0),
        };
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
                    .image_info(slice::from_ref(&cubemap.info))
                    .build()],
            );
            self.device.cmd_push_constants(
                cmd_buffer,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&params),
            );
            self.device.cmd_draw(cmd_buffer, 4, 1, 0, 0);
            self.device.debug(|d| d.cmd_end_label(cmd_buffer));
        }

        Ok(DrawPayload::new(engine.end_secondary_draw_commands(cmd_buffer)?))
    }

    pub fn rebuild_pipeline(&mut self, engine: &VulkanEngine) -> VulkanResult<()> {
        let pipeline = Pipeline::builder_graphics(&self.shader)
            .descriptor_layout(&self.desc_layout)
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
            self.pipeline.cleanup(&self.device);
            self.shader.cleanup(&self.device);
            self.desc_layout.cleanup(&self.device);
            self.cmd_buffers.cleanup(&self.device);
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
struct SkyboxParams {
    viewproj_inv: Mat4,
    lod: Vec4,
}
