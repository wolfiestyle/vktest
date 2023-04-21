use crate::create::CreateFromInfo;
use crate::device::{ImageParams, VulkanDevice};
use crate::engine::{SamplerOptions, Texture, VulkanEngine};
use crate::pipeline::Pipeline;
use crate::types::*;
use ash::vk;
use inline_spirv::include_spirv;
use std::slice;
use std::sync::Arc;

pub struct Baker {
    device: Arc<VulkanDevice>,
    irrmap_pipeline: Pipeline,
    prefilter_pipeline: Pipeline,
    brdf_pipeline: Pipeline,
}

const IRRMAP_SIZE: u32 = 32;
const IRRMAP_WG: u32 = IRRMAP_SIZE / 16;
const PREFILTERED_SIZE: u32 = 256;
const PREFILTERED_WG: u32 = PREFILTERED_SIZE / 8;
const BRDFLUT_SIZE: u32 = 512;
const BRDFLUT_WG: u32 = BRDFLUT_SIZE / 16;

impl Baker {
    pub fn new(engine: &VulkanEngine) -> VulkanResult<Self> {
        let device = engine.device.clone();

        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
        ];
        // irradiance map (diffuse lighting)
        let irrmap_shader = vk::ShaderModuleCreateInfo::builder()
            .code(include_spirv!("src/shaders/irrmap.comp.glsl", comp, glsl))
            .create(&device)?;
        let desc_layout = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&bindings)
            .create(&device)?;
        let irrmap_pipeline = Pipeline::builder_compute(irrmap_shader)
            .descriptor_layout(&desc_layout)
            .build(engine)?;
        // prefilter map (specular lighting)
        let prefilter_shader = vk::ShaderModuleCreateInfo::builder()
            .code(include_spirv!("src/shaders/prefilter.comp.glsl", comp, glsl))
            .create(&device)?;
        let desc_layout = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&bindings)
            .create(&device)?;
        let push_constants = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<f32>() as _,
        };
        let prefilter_pipeline = Pipeline::builder_compute(prefilter_shader)
            .descriptor_layout(&desc_layout)
            .push_constants(&[push_constants])
            .build(engine)?;
        // precomputed BRDF for specular
        let brdf_shader = vk::ShaderModuleCreateInfo::builder()
            .code(include_spirv!("src/shaders/brdf_lut.comp.glsl", comp, glsl))
            .create(&device)?;
        let desc_layout = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&[vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build()])
            .create(&device)?;
        let brdf_pipeline = Pipeline::builder_compute(brdf_shader)
            .descriptor_layout(&desc_layout)
            .build(engine)?;
        unsafe {
            device.destroy_shader_module(irrmap_shader, None);
            device.destroy_shader_module(prefilter_shader, None);
            device.destroy_shader_module(brdf_shader, None);
        }
        Ok(Self {
            device,
            irrmap_pipeline,
            prefilter_pipeline,
            brdf_pipeline,
        })
    }

    pub fn generate_irradiance_map(&self, cubemap: &Texture) -> VulkanResult<Texture> {
        let params = ImageParams {
            width: IRRMAP_SIZE,
            height: IRRMAP_SIZE,
            layers: 6,
            format: vk::Format::R16G16B16A16_SFLOAT,
            ..Default::default()
        };
        let mut irrmap = Texture::new_empty(
            &self.device,
            params,
            vk::ImageCreateFlags::CUBE_COMPATIBLE,
            vk::ImageLayout::GENERAL,
            cubemap.sampler,
        )?;
        let cmd_buffer = self.device.begin_one_time_commands()?;
        unsafe {
            let cubemap_info = cubemap.descriptor();
            let irrmap_info = irrmap.descriptor();
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, *self.irrmap_pipeline);
            self.device.pushdesc_fn.cmd_push_descriptor_set(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.irrmap_pipeline.layout,
                0,
                &[
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(slice::from_ref(&cubemap_info))
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(slice::from_ref(&irrmap_info))
                        .build(),
                ],
            );
            self.device.cmd_dispatch(cmd_buffer, IRRMAP_WG, IRRMAP_WG, 6);
        }
        irrmap.transition_layout(&self.device, cmd_buffer, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        self.device.end_one_time_commands(cmd_buffer)?;
        Ok(irrmap)
    }

    pub fn generate_prefilter_map(&self, cubemap: &Texture) -> VulkanResult<Texture> {
        let mip_levels = PREFILTERED_WG.ilog2() + 1;
        let params = ImageParams {
            width: PREFILTERED_SIZE,
            height: PREFILTERED_SIZE,
            layers: 6,
            mip_levels,
            format: vk::Format::R16G16B16A16_SFLOAT,
            ..Default::default()
        };
        let mut prefmap = Texture::new_empty(
            &self.device,
            params,
            vk::ImageCreateFlags::CUBE_COMPATIBLE,
            vk::ImageLayout::GENERAL,
            cubemap.sampler,
        )?;
        let mip_imgviews = (0..mip_levels)
            .map(|level| {
                prefmap.image.create_view_subresource(
                    &self.device,
                    vk::ImageViewType::CUBE,
                    vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: level,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 6,
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let cmd_buffer = self.device.begin_one_time_commands()?;
        unsafe {
            let cubemap_info = cubemap.descriptor();
            let prefilter_info = prefmap.descriptor();
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, *self.prefilter_pipeline);
            self.device.pushdesc_fn.cmd_push_descriptor_set(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.prefilter_pipeline.layout,
                0,
                &[vk::WriteDescriptorSet::builder()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(slice::from_ref(&cubemap_info))
                    .build()],
            );
            let mut wg_size = PREFILTERED_WG;
            for mip in 0..mip_levels {
                let roughness = mip as f32 / (mip_levels - 1) as f32;
                //let v = (roughness.exp() - 1.0) / (std::f32::consts::E - 1.0);
                let mip_info = vk::DescriptorImageInfo {
                    image_view: mip_imgviews[mip as usize],
                    ..prefilter_info
                };
                self.device.pushdesc_fn.cmd_push_descriptor_set(
                    cmd_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.prefilter_pipeline.layout,
                    0,
                    &[vk::WriteDescriptorSet::builder()
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(slice::from_ref(&mip_info))
                        .build()],
                );
                self.device.cmd_push_constants(
                    cmd_buffer,
                    self.prefilter_pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&roughness),
                );
                self.device.cmd_dispatch(cmd_buffer, wg_size, wg_size, 6);
                wg_size /= 2;
            }
        }
        prefmap.transition_layout(&self.device, cmd_buffer, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        self.device.end_one_time_commands(cmd_buffer)?;
        self.device.dispose_of(mip_imgviews);
        Ok(prefmap)
    }

    pub fn generate_brdf_lut(&self, engine: &VulkanEngine) -> VulkanResult<Texture> {
        let params = ImageParams {
            width: BRDFLUT_SIZE,
            height: BRDFLUT_SIZE,
            format: vk::Format::R16G16_SFLOAT,
            ..Default::default()
        };
        let sampler = engine.get_sampler(SamplerOptions {
            wrap_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            wrap_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            ..Default::default()
        })?;
        let mut brdf_lut = Texture::new_empty(
            &self.device,
            params,
            vk::ImageCreateFlags::empty(),
            vk::ImageLayout::GENERAL,
            sampler,
        )?;
        let cmd_buffer = self.device.begin_one_time_commands()?;
        unsafe {
            let bdrf_info = brdf_lut.descriptor();
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, *self.brdf_pipeline);
            self.device.pushdesc_fn.cmd_push_descriptor_set(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.brdf_pipeline.layout,
                0,
                &[vk::WriteDescriptorSet::builder()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(slice::from_ref(&bdrf_info))
                    .build()],
            );
            self.device.cmd_dispatch(cmd_buffer, BRDFLUT_WG, BRDFLUT_WG, 1);
        }
        brdf_lut.transition_layout(&self.device, cmd_buffer, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        self.device.end_one_time_commands(cmd_buffer)?;
        Ok(brdf_lut)
    }
}

impl Drop for Baker {
    fn drop(&mut self) {
        unsafe {
            self.irrmap_pipeline.cleanup(&self.device);
            self.prefilter_pipeline.cleanup(&self.device);
            self.brdf_pipeline.cleanup(&self.device);
        }
    }
}
