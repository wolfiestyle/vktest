use crate::create::CreateFromInfo;
use crate::device::ImageParams;
use crate::device::VulkanDevice;
use crate::engine::{Texture, VulkanEngine};
use crate::pipeline::Pipeline;
use crate::types::*;
use ash::vk;
use inline_spirv::include_spirv;
use std::slice;
use std::sync::Arc;

pub struct Baker {
    device: Arc<VulkanDevice>,
    irrmap_pipeline: Pipeline,
}

impl Baker {
    pub fn new(engine: &VulkanEngine) -> VulkanResult<Self> {
        let device = engine.device.clone();
        let shader = vk::ShaderModuleCreateInfo::builder()
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
        let irrmap_pipeline = Pipeline::builder_compute(shader).descriptor_layout(&desc_layout).build(engine)?;
        unsafe {
            device.destroy_shader_module(shader, None);
        }
        Ok(Self { device, irrmap_pipeline })
    }

    pub fn generate_irradiance_map(&self, cubemap: &Texture) -> VulkanResult<Texture> {
        let params = ImageParams {
            width: 32,
            height: 32,
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
            self.device.cmd_dispatch(cmd_buffer, 1, 1, 6);
        }
        irrmap.transition_layout(&self.device, cmd_buffer, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        self.device.end_one_time_commands(cmd_buffer)?;
        Ok(irrmap)
    }
}

impl Drop for Baker {
    fn drop(&mut self) {
        unsafe {
            self.irrmap_pipeline.cleanup(&self.device);
        }
    }
}
