use crate::create::CreateFromInfo;
use crate::device::VulkanDevice;
use crate::engine::{Shader, VulkanEngine};
use crate::swapchain::Swapchain;
use crate::types::*;
use crate::vertex::VertexInput;
use ash::vk;
use cstr::cstr;
use std::slice;

#[derive(Debug)]
pub struct Pipeline {
    pub(crate) handle: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
}

impl Pipeline {
    #[inline]
    pub fn builder_graphics(shader: &Shader) -> GraphicsPipelineBuilder {
        GraphicsPipelineBuilder::new(shader)
    }

    #[inline]
    pub fn builder_compute<'a>(shader: vk::ShaderModule) -> ComputePipelineBuilder<'a> {
        ComputePipelineBuilder::new(shader)
    }

    fn create_graphics_pipeline(
        engine: &VulkanEngine, layout: vk::PipelineLayout, params: GraphicsPipelineBuilder,
    ) -> VulkanResult<vk::Pipeline> {
        let device = &engine.device;
        let entry_point = cstr!("main");
        let spec_info = vk::SpecializationInfo::builder()
            .map_entries(params.spec_entries)
            .data(params.spec_data);
        let shader_stages_ci = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(params.shader.vert)
                .name(entry_point)
                .specialization_info(&spec_info)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(params.shader.frag)
                .name(entry_point)
                .specialization_info(&spec_info)
                .build(),
        ];

        let vertex_input_ci = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&params.binding_desc)
            .vertex_attribute_descriptions(&params.attrib_desc);

        let input_assembly_ci = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(params.topology)
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
            .cull_mode(params.mode.cull_mode())
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

        let multisample_ci = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(engine.swapchain.samples)
            .min_sample_shading(1.0);

        let color_attach = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(params.mode.blend_enable())
            .src_color_blend_factor(vk::BlendFactor::ONE) // premultiplied alpha
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blend_ci = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(slice::from_ref(&color_attach));

        let depth_stencil_ci = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(params.mode.depth_test())
            .depth_write_enable(params.mode.depth_write())
            .depth_compare_op(params.mode.depth_compare_op())
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);

        let mut pipeline_rendering_ci = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(slice::from_ref(&params.color_format))
            .depth_attachment_format(params.depth_format);

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
            .layout(layout)
            .push_next(&mut pipeline_rendering_ci);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(engine.pipeline_cache, slice::from_ref(&pipeline_ci), None)
                .map_err(|(_, err)| VkError::VulkanMsg("Error creating pipeline", err))?
        };

        Ok(pipeline[0])
    }

    fn create_compute_pipeline(
        engine: &VulkanEngine, layout: vk::PipelineLayout, params: ComputePipelineBuilder,
    ) -> VulkanResult<vk::Pipeline> {
        let entry_point = cstr!("main");
        let spec_info = vk::SpecializationInfo::builder()
            .map_entries(params.spec_entries)
            .data(params.spec_data);
        let shader_stages_ci = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(params.shader)
            .name(entry_point)
            .specialization_info(&spec_info)
            .build();

        let pipeline_ci = vk::ComputePipelineCreateInfo::builder().stage(shader_stages_ci).layout(layout);

        let pipeline = unsafe {
            engine
                .device
                .create_compute_pipelines(engine.pipeline_cache, slice::from_ref(&pipeline_ci), None)
                .map_err(|(_, err)| VkError::VulkanMsg("Error creating compute pipeline", err))?
        };

        Ok(pipeline[0])
    }
}

impl Cleanup<VulkanDevice> for Pipeline {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_pipeline(self.handle, None);
        device.destroy_pipeline_layout(self.layout, None);
    }
}

impl std::ops::Deref for Pipeline {
    type Target = vk::Pipeline;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

#[derive(Debug, Clone)]
pub struct GraphicsPipelineBuilder<'a> {
    pub shader: &'a Shader,
    pub desc_layouts: &'a [vk::DescriptorSetLayout],
    pub push_constants: &'a [vk::PushConstantRange],
    pub binding_desc: Vec<vk::VertexInputBindingDescription>,
    pub attrib_desc: Vec<vk::VertexInputAttributeDescription>,
    pub color_format: vk::Format,
    pub depth_format: vk::Format,
    pub mode: PipelineMode,
    pub topology: vk::PrimitiveTopology,
    pub spec_entries: &'a [vk::SpecializationMapEntry],
    pub spec_data: &'a [u8],
}

impl<'a> GraphicsPipelineBuilder<'a> {
    pub fn new(shader: &'a Shader) -> Self {
        Self {
            shader,
            desc_layouts: &[],
            push_constants: &[],
            binding_desc: vec![],
            attrib_desc: vec![],
            color_format: vk::Format::UNDEFINED,
            depth_format: vk::Format::UNDEFINED,
            mode: PipelineMode::Opaque,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            spec_entries: &[],
            spec_data: &[],
        }
    }

    pub fn vertex_input<V: VertexInput>(mut self) -> Self {
        self.binding_desc = vec![V::binding_desc(0)];
        self.attrib_desc = V::attr_desc(0);
        self
    }

    pub fn descriptor_layout(mut self, set_layout: &'a vk::DescriptorSetLayout) -> Self {
        self.desc_layouts = slice::from_ref(set_layout);
        self
    }

    pub fn descriptor_layouts(mut self, set_layouts: &'a [vk::DescriptorSetLayout]) -> Self {
        self.desc_layouts = set_layouts;
        self
    }

    pub fn push_constants(mut self, push_constants: &'a [vk::PushConstantRange]) -> Self {
        self.push_constants = push_constants;
        self
    }

    pub fn render_to_swapchain(mut self, swapchain: &Swapchain) -> Self {
        self.color_format = swapchain.format;
        self.depth_format = swapchain.depth_format;
        self
    }

    pub fn mode(mut self, mode: PipelineMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn topology(mut self, topology: vk::PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    pub fn spec_constants(mut self, entries: &'a [vk::SpecializationMapEntry], data: &'a [u8]) -> Self {
        self.spec_entries = entries;
        self.spec_data = data;
        self
    }

    pub fn build(self, engine: &VulkanEngine) -> VulkanResult<Pipeline> {
        let layout = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&self.desc_layouts)
            .push_constant_ranges(self.push_constants)
            .create(&engine.device)?;
        let handle = Pipeline::create_graphics_pipeline(engine, layout, self)?;
        Ok(Pipeline { handle, layout })
    }
}

pub struct ComputePipelineBuilder<'a> {
    pub shader: vk::ShaderModule,
    pub desc_layouts: &'a [vk::DescriptorSetLayout],
    pub push_constants: &'a [vk::PushConstantRange],
    pub spec_entries: &'a [vk::SpecializationMapEntry],
    pub spec_data: &'a [u8],
}

impl<'a> ComputePipelineBuilder<'a> {
    pub fn new(shader: vk::ShaderModule) -> Self {
        Self {
            shader,
            desc_layouts: &[],
            push_constants: &[],
            spec_entries: &[],
            spec_data: &[],
        }
    }

    pub fn descriptor_layout(mut self, set_layout: &'a vk::DescriptorSetLayout) -> Self {
        self.desc_layouts = slice::from_ref(set_layout);
        self
    }

    pub fn descriptor_layouts(mut self, set_layouts: &'a [vk::DescriptorSetLayout]) -> Self {
        self.desc_layouts = set_layouts;
        self
    }

    pub fn push_constants(mut self, push_constants: &'a [vk::PushConstantRange]) -> Self {
        self.push_constants = push_constants;
        self
    }

    pub fn spec_constants(mut self, entries: &'a [vk::SpecializationMapEntry], data: &'a [u8]) -> Self {
        self.spec_entries = entries;
        self.spec_data = data;
        self
    }

    pub fn build(self, engine: &VulkanEngine) -> VulkanResult<Pipeline> {
        let layout = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&self.desc_layouts)
            .push_constant_ranges(self.push_constants)
            .create(&engine.device)?;
        let handle = Pipeline::create_compute_pipeline(engine, layout, self)?;
        Ok(Pipeline { handle, layout })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PipelineMode {
    Opaque,
    Transparent,
    Background,
    Overlay,
}

impl PipelineMode {
    fn depth_test(self) -> bool {
        !matches!(self, Self::Overlay)
    }

    fn depth_write(self) -> bool {
        matches!(self, Self::Opaque)
    }

    fn depth_compare_op(self) -> vk::CompareOp {
        match self {
            Self::Opaque | Self::Transparent => vk::CompareOp::LESS,
            Self::Background => vk::CompareOp::EQUAL,
            Self::Overlay => vk::CompareOp::ALWAYS,
        }
    }

    fn cull_mode(self) -> vk::CullModeFlags {
        match self {
            Self::Background | Self::Overlay => vk::CullModeFlags::NONE,
            _ => vk::CullModeFlags::BACK,
        }
    }

    fn blend_enable(self) -> bool {
        matches!(self, Self::Transparent | Self::Overlay)
    }
}
