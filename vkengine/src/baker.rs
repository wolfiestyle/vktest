use crate::create::CreateFromInfo;
use crate::device::{ImageParams, VulkanDevice};
use crate::engine::{Texture, VulkanEngine};
use crate::pipeline::Pipeline;
use crate::types::*;
use ash::vk;
use inline_spirv::include_spirv;
use std::mem::size_of;
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
const PREFILTERED_MIP_LEVELS: u32 = PREFILTERED_SIZE.ilog2() + 1;
const BRDFLUT_SIZE: u32 = 512;
const BRDFLUT_WG: u32 = BRDFLUT_SIZE / 16;

impl Baker {
    pub fn new(engine: &VulkanEngine) -> VulkanResult<Self> {
        let device = engine.device.clone();

        // irradiance map (diffuse lighting)
        let irrmap_shader = vk::ShaderModuleCreateInfo::builder()
            .code(include_spirv!("src/shaders/irrmap.comp.glsl", comp, glsl))
            .create(&device)?;
        let desc_layout = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&[
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
            ])
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
            .bindings(&[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(PREFILTERED_MIP_LEVELS)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ])
            .create(&device)?;
        let push_constants = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: size_of::<u32>() as _,
        };
        let spec_constants = vk::SpecializationMapEntry {
            constant_id: 1,
            offset: 0,
            size: size_of::<u32>(),
        };
        let prefilter_pipeline = Pipeline::builder_compute(prefilter_shader)
            .descriptor_layout(&desc_layout)
            .push_constants(&[push_constants])
            .spec_constants(slice::from_ref(&spec_constants), bytemuck::bytes_of(&PREFILTERED_MIP_LEVELS))
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
            vk::Sampler::null(),
        )?;
        let cmd_buffer = self.device.begin_one_time_commands()?;
        unsafe {
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
                        .image_info(slice::from_ref(&cubemap.info))
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(slice::from_ref(&irrmap.info))
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
        let params = ImageParams {
            width: PREFILTERED_SIZE,
            height: PREFILTERED_SIZE,
            layers: 6,
            mip_levels: PREFILTERED_MIP_LEVELS,
            format: vk::Format::R16G16B16A16_SFLOAT,
            ..Default::default()
        };
        let mut prefmap = Texture::new_empty(
            &self.device,
            params,
            vk::ImageCreateFlags::CUBE_COMPATIBLE,
            vk::ImageLayout::GENERAL,
            vk::Sampler::null(),
        )?;
        let mip_infos = (0..PREFILTERED_MIP_LEVELS)
            .map(|level| {
                let image_view = prefmap.image.create_view_subresource(
                    &self.device,
                    vk::ImageViewType::CUBE,
                    vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: level,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 6,
                    },
                )?;
                Ok(vk::DescriptorImageInfo {
                    image_view,
                    ..prefmap.info
                })
            })
            .collect::<Result<Vec<_>, VkError>>()?;
        let cmd_buffer = self.device.begin_one_time_commands()?;
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, *self.prefilter_pipeline);
            self.device.pushdesc_fn.cmd_push_descriptor_set(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.prefilter_pipeline.layout,
                0,
                &[
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(slice::from_ref(&cubemap.info))
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&mip_infos)
                        .build(),
                ],
            );
            let mut wg_size = PREFILTERED_WG;
            for level in 0..PREFILTERED_MIP_LEVELS {
                self.device.cmd_push_constants(
                    cmd_buffer,
                    self.prefilter_pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&level),
                );
                self.device.cmd_dispatch(cmd_buffer, wg_size, wg_size, 6);
                wg_size = 1.max(wg_size / 2);
            }
        }
        prefmap.transition_layout(&self.device, cmd_buffer, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        self.device.end_one_time_commands(cmd_buffer)?;
        for info in mip_infos {
            self.device.dispose_of(info.image_view);
        }
        Ok(prefmap)
    }

    pub fn generate_brdf_lut(&self) -> VulkanResult<Texture> {
        let params = ImageParams {
            width: BRDFLUT_SIZE,
            height: BRDFLUT_SIZE,
            format: vk::Format::R16G16_SFLOAT,
            ..Default::default()
        };
        let mut brdf_lut = Texture::new_empty(
            &self.device,
            params,
            vk::ImageCreateFlags::empty(),
            vk::ImageLayout::GENERAL,
            vk::Sampler::null(),
        )?;
        let cmd_buffer = self.device.begin_one_time_commands()?;
        unsafe {
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
                    .image_info(slice::from_ref(&brdf_lut.info))
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
