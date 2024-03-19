use crate::create::CreateFromInfo;
use crate::device::{ImageParams, VulkanDevice};
use crate::engine::{SamplerOptions, VulkanEngine};
use crate::pipeline::Pipeline;
use crate::texture::Texture;
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
    eq2cube_pipeline: Pipeline,
}

const IRRMAP_SIZE: u32 = 32;
const IRRMAP_WG: u32 = IRRMAP_SIZE / 16;
const PREFILTERED_SIZE: u32 = 256;
const PREFILTERED_WG: u32 = PREFILTERED_SIZE / 8;
const PREFILTERED_MIP_LEVELS: u32 = PREFILTERED_SIZE.ilog2() + 1;
const BRDFLUT_SIZE: u32 = 256;
const BRDFLUT_WG: u32 = BRDFLUT_SIZE / 16;
const EQ2CUBE_WG_SIZE: u32 = 16;

impl Baker {
    pub fn new(engine: &VulkanEngine) -> VulkanResult<Self> {
        Ok(Self {
            device: engine.device.clone(),
            irrmap_pipeline: Self::create_irrmap_pipeline(engine)?,
            prefilter_pipeline: Self::create_prefilter_pipeline(engine)?,
            brdf_pipeline: Self::create_brdf_lut_pipeline(engine)?,
            eq2cube_pipeline: Self::create_eq2cube_pipeline(engine)?,
        })
    }

    fn create_irrmap_pipeline(engine: &VulkanEngine) -> VulkanResult<Pipeline> {
        let device = &*engine.device;
        let irrmap_shader = vk::ShaderModuleCreateInfo::builder()
            .code(include_spirv!("src/shaders/irrmap.comp.glsl", comp, glsl))
            .create(&device)?;
        let sampler = engine.get_sampler(vk::SamplerAddressMode::CLAMP_TO_EDGE.into())?;
        let desc_layout = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .immutable_samplers(slice::from_ref(&sampler))
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
        unsafe {
            device.destroy_shader_module(irrmap_shader, None);
        }
        Ok(irrmap_pipeline)
    }

    fn create_prefilter_pipeline(engine: &VulkanEngine) -> VulkanResult<Pipeline> {
        let device = &*engine.device;
        let prefilter_shader = vk::ShaderModuleCreateInfo::builder()
            .code(include_spirv!("src/shaders/prefilter.comp.glsl", comp, glsl))
            .create(&device)?;
        let sampler = engine.get_sampler(vk::SamplerAddressMode::CLAMP_TO_EDGE.into())?;
        let desc_layout = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .immutable_samplers(slice::from_ref(&sampler))
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
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
        unsafe {
            device.destroy_shader_module(prefilter_shader, None);
        }
        Ok(prefilter_pipeline)
    }

    fn create_brdf_lut_pipeline(engine: &VulkanEngine) -> VulkanResult<Pipeline> {
        let device = &*engine.device;
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
            device.destroy_shader_module(brdf_shader, None);
        }
        Ok(brdf_pipeline)
    }

    fn create_eq2cube_pipeline(engine: &VulkanEngine) -> VulkanResult<Pipeline> {
        let device = &*engine.device;
        let eq2cube_shader = vk::ShaderModuleCreateInfo::builder()
            .code(include_spirv!("src/shaders/equirect2cube.comp.glsl", comp, glsl))
            .create(&device)?;
        let sampler = engine.get_sampler(SamplerOptions {
            wrap_u: vk::SamplerAddressMode::REPEAT,
            wrap_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            ..Default::default()
        })?;
        let desc_layout = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .immutable_samplers(slice::from_ref(&sampler))
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ])
            .create(&device)?;
        let eq2cube_pipeline = Pipeline::builder_compute(eq2cube_shader)
            .descriptor_layout(&desc_layout)
            .build(engine)?;
        unsafe {
            device.destroy_shader_module(eq2cube_shader, None);
        }
        Ok(eq2cube_pipeline)
    }

    pub fn generate_irradiance_map(&self, cubemap: &Texture) -> VulkanResult<Texture> {
        eprintln!("generating irradiance map..");
        let params = ImageParams {
            width: IRRMAP_SIZE,
            height: IRRMAP_SIZE,
            layers: 6,
            format: vk::Format::R16G16B16A16_SFLOAT,
            ..Default::default()
        };
        let mut irrmap = Texture::new_empty(&self.device, params, vk::ImageCreateFlags::CUBE_COMPATIBLE, vk::Sampler::null())?;
        self.device
            .debug(|d| d.set_object_name(&self.device, &*irrmap.image, "Irradiance map"));
        let cmd_buffer = self.device.begin_one_time_commands()?;
        irrmap.transition_layout(&self.device, cmd_buffer, vk::ImageLayout::GENERAL);
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
        eprintln!("generating prefiltered map..");
        let params = ImageParams {
            width: PREFILTERED_SIZE,
            height: PREFILTERED_SIZE,
            layers: 6,
            mip_levels: PREFILTERED_MIP_LEVELS,
            format: vk::Format::R16G16B16A16_SFLOAT,
            ..Default::default()
        };
        let mut prefmap = Texture::new_empty(&self.device, params, vk::ImageCreateFlags::CUBE_COMPATIBLE, vk::Sampler::null())?;
        self.device
            .debug(|d| d.set_object_name(&self.device, &*prefmap.image, "Prefiltered map"));
        let cmd_buffer = self.device.begin_one_time_commands()?;
        prefmap.transition_layout(&self.device, cmd_buffer, vk::ImageLayout::GENERAL);
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
        unsafe {
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
                    .image_info(slice::from_ref(&cubemap.info))
                    .build()],
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
                self.device.pushdesc_fn.cmd_push_descriptor_set(
                    cmd_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.prefilter_pipeline.layout,
                    0,
                    &[vk::WriteDescriptorSet::builder()
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(slice::from_ref(&mip_infos[level as usize]))
                        .build()],
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
        eprintln!("generating BRDF lut..");
        let params = ImageParams {
            width: BRDFLUT_SIZE,
            height: BRDFLUT_SIZE,
            format: vk::Format::R16G16_UNORM,
            ..Default::default()
        };
        let mut brdf_lut = Texture::new_empty(&self.device, params, vk::ImageCreateFlags::empty(), vk::Sampler::null())?;
        self.device.debug(|d| d.set_object_name(&self.device, &*brdf_lut.image, "BRDF lut"));
        let cmd_buffer = self.device.begin_one_time_commands()?;
        brdf_lut.transition_layout(&self.device, cmd_buffer, vk::ImageLayout::GENERAL);
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

    pub fn equirect_to_cubemap(&self, equirect: &Texture, gen_mipmaps: bool) -> VulkanResult<Texture> {
        eprintln!("converting equirect to cubemap..");
        let size = equirect.image.props.size().min_element();
        let params = ImageParams {
            width: size,
            height: size,
            layers: 6,
            mip_levels: if gen_mipmaps { size.ilog2() + 1 } else { 1 },
            format: vk::Format::R16G16B16A16_SFLOAT,
            ..Default::default()
        };
        let mut cubemap = Texture::new_empty(&self.device, params, vk::ImageCreateFlags::CUBE_COMPATIBLE, vk::Sampler::null())?;
        self.device
            .debug(|d| d.set_object_name(&self.device, &*cubemap.image, "Cubemap image"));
        let cmd_buffer = self.device.begin_one_time_commands()?;
        cubemap.transition_layout(&self.device, cmd_buffer, vk::ImageLayout::GENERAL);
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, *self.eq2cube_pipeline);
            self.device.pushdesc_fn.cmd_push_descriptor_set(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.eq2cube_pipeline.layout,
                0,
                &[
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(slice::from_ref(&equirect.info))
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(slice::from_ref(&cubemap.info))
                        .build(),
                ],
            );
            let wg_size = size / EQ2CUBE_WG_SIZE;
            self.device.cmd_dispatch(cmd_buffer, wg_size, wg_size, 6);
        }
        if gen_mipmaps {
            cubemap.transition_layout(&self.device, cmd_buffer, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            unsafe {
                self.device.generate_mipmaps(cmd_buffer, *cubemap.image, params);
            }
            cubemap.info.image_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        } else {
            cubemap.transition_layout(&self.device, cmd_buffer, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
        self.device.end_one_time_commands(cmd_buffer)?;
        Ok(cubemap)
    }
}

impl Drop for Baker {
    fn drop(&mut self) {
        unsafe {
            self.irrmap_pipeline.cleanup(&self.device);
            self.prefilter_pipeline.cleanup(&self.device);
            self.brdf_pipeline.cleanup(&self.device);
            self.eq2cube_pipeline.cleanup(&self.device);
        }
    }
}
