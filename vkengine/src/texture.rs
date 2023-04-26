use crate::device::{ImageData, ImageParams, VkImage, VulkanDevice};
use crate::types::*;
use ash::vk;
use glam::UVec2;

#[derive(Debug, Clone, Copy, Default)]
pub struct TextureOptions {
    pub gen_mipmaps: bool,
    pub swizzle: Option<vk::ComponentMapping>,
}

#[derive(Debug)]
pub struct Texture {
    pub image: VkImage,
    pub info: vk::DescriptorImageInfo,
}

impl Texture {
    pub fn new(
        device: &VulkanDevice, size: UVec2, format: vk::Format, data: ImageData, sampler: vk::Sampler, options: TextureOptions,
    ) -> VulkanResult<Self> {
        let params = ImageParams {
            width: size.x,
            height: size.y,
            format,
            layers: data.layer_count(),
            mip_levels: if options.gen_mipmaps { size.max_element().ilog2() + 1 } else { 1 },
            ..Default::default()
        };
        let image = device.create_image_from_data(params, data, data.image_create_flags())?;
        let image_view = if let Some(swizzle) = options.swizzle {
            image.create_view_swizzle(device, data.view_type(), swizzle)?
        } else {
            image.create_view(device, data.view_type())?
        };
        Ok(Self {
            image,
            info: vk::DescriptorImageInfo {
                sampler,
                image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        })
    }

    pub fn new_empty(device: &VulkanDevice, params: ImageParams, flags: vk::ImageCreateFlags, sampler: vk::Sampler) -> VulkanResult<Self> {
        let image = device.allocate_image(
            params,
            flags,
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
            "Texture image",
        )?;
        let view_type = if flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE) {
            vk::ImageViewType::CUBE
        } else {
            vk::ImageViewType::TYPE_2D
        };
        let image_view = image.create_view(device, view_type)?;
        Ok(Self {
            image,
            info: vk::DescriptorImageInfo {
                sampler,
                image_view,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
        })
    }

    #[allow(unused_assignments)]
    pub fn from_dynamicimage(
        device: &VulkanDevice, image: &gltf_import::DynamicImage, is_srgb: bool, gen_mipmaps: bool, sampler: vk::Sampler,
    ) -> VulkanResult<Texture> {
        use gltf_import::DynamicImage::*;

        let mut temp8 = vec![];
        let mut temp16 = vec![];
        let mut temp32 = vec![];

        let bytes = match image {
            ImageLuma8(img) => img.as_raw(),
            ImageLumaA8(img) => img.as_raw(),
            ImageRgb8(_) => {
                temp8 = image.to_rgba8().into_raw();
                &temp8
            }
            ImageRgba8(img) => img.as_raw(),
            ImageLuma16(img) => bytemuck::cast_slice(img.as_raw()),
            ImageLumaA16(img) => bytemuck::cast_slice(img.as_raw()),
            ImageRgb16(_) => {
                temp16 = image.to_rgba16().into_raw();
                bytemuck::cast_slice(&temp16)
            }
            ImageRgba16(img) => bytemuck::cast_slice(img.as_raw()),
            ImageRgb32F(_) => {
                temp32 = image.to_rgba32f().into_raw();
                bytemuck::cast_slice(&temp32)
            }
            ImageRgba32F(img) => bytemuck::cast_slice(img.as_raw()),
            _ => {
                temp8 = image.to_rgba8().into_raw();
                &temp8
            }
        };

        let format = match image {
            ImageLuma8(_) if is_srgb => vk::Format::R8_SRGB,
            ImageLuma8(_) if !is_srgb => vk::Format::R8_UNORM,
            ImageLumaA8(_) if is_srgb => vk::Format::R8G8_SRGB,
            ImageLumaA8(_) if !is_srgb => vk::Format::R8G8_UNORM,
            ImageRgb8(_) | ImageRgba8(_) if is_srgb => vk::Format::R8G8B8A8_SRGB,
            ImageRgb8(_) | ImageRgba8(_) if !is_srgb => vk::Format::R8G8B8A8_UNORM,
            ImageLuma16(_) => vk::Format::R16_UNORM,
            ImageLumaA16(_) => vk::Format::R16G16_UNORM,
            ImageRgb16(_) | ImageRgba16(_) => vk::Format::R16G16B16A16_UNORM,
            ImageRgb32F(_) | ImageRgba32F(_) => vk::Format::R32G32B32A32_SFLOAT,
            _ => {
                if is_srgb {
                    vk::Format::R8G8B8A8_SRGB
                } else {
                    vk::Format::R8G8B8A8_UNORM
                }
            }
        };

        let swizzle = match image {
            ImageLuma8(_) | ImageLuma16(_) => Some(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::R,
                b: vk::ComponentSwizzle::R,
                a: vk::ComponentSwizzle::ONE,
            }),
            ImageLumaA8(_) | ImageLumaA16(_) => Some(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::R,
                b: vk::ComponentSwizzle::R,
                a: vk::ComponentSwizzle::G,
            }),
            _ => None,
        };

        Texture::new(
            device,
            [image.width(), image.height()].into(),
            format,
            ImageData::Single(bytes),
            sampler,
            TextureOptions { gen_mipmaps, swizzle },
        )
    }

    pub fn update(&mut self, device: &VulkanDevice, pos: UVec2, size: UVec2, data: &[u8]) -> VulkanResult<()> {
        device.update_image_from_data(&self.image, pos, size, 0, ImageData::Single(data))
    }

    pub fn transition_layout(&mut self, device: &VulkanDevice, cmd_buffer: vk::CommandBuffer, new_layout: vk::ImageLayout) {
        device.transition_image_layout(
            cmd_buffer,
            *self.image,
            self.image.props.subresource_range(),
            self.info.image_layout,
            new_layout,
        );
        self.info.image_layout = new_layout;
    }
}

impl Cleanup<VulkanDevice> for Texture {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        self.info.image_view.cleanup(device);
        self.image.cleanup(device);
    }
}
