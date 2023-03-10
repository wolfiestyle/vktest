use crate::camera::Camera;
use crate::device::{ImageData, Swapchain, VkImage, VulkanDevice};
use crate::instance::DeviceSelection;
use crate::types::*;
use ash::vk;
use cstr::cstr;
use glam::{UVec2, Vec3};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::collections::HashMap;
use std::slice;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thread_local::ThreadLocal;

const SWAPCHAIN_IMAGE_COUNT: u32 = 3;
const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanEngine {
    pub(crate) device: Arc<VulkanDevice>,
    window_size: UVec2,
    window_resized: bool,
    pub(crate) swapchain: Swapchain,
    thread_cmd_pools: ThreadLocal<vk::CommandPool>,
    main_cmd_buffers: Vec<vk::CommandBuffer>,
    frame_state: Vec<FrameState>,
    current_frame: u64,
    samplers: Mutex<HashMap<SamplerOptions, vk::Sampler>>,
    pub(crate) default_texture: Texture,
    pub(crate) pipeline_cache: vk::PipelineCache,
    prev_frame_time: Instant,
    last_frame_time: Instant,
    pub camera: Camera,
    pub sunlight: Vec3,
}

impl VulkanEngine {
    pub fn new<W>(window: &W, app_name: &str, device_selection: DeviceSelection) -> VulkanResult<Self>
    where
        W: HasRawDisplayHandle + HasRawWindowHandle + WindowSize,
    {
        let device = VulkanDevice::new(window, app_name, device_selection)?;
        let depth_format = device.find_depth_format(false)?;
        let msaa_samples = device.dev_info.get_max_samples();
        let window_size = window.window_size().into();
        let swapchain = Swapchain::new(&device, window_size, SWAPCHAIN_IMAGE_COUNT, depth_format, msaa_samples)?;
        eprintln!("color_format: {:?}, depth_format: {depth_format:?}", swapchain.format);

        let frame_state = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| FrameState::new(&device))
            .collect::<Result<Vec<_>, _>>()?;

        let pixel = [255, 255, 255, 255];
        let default_texture = Texture::new(&device, 1, 1, vk::Format::R8G8B8A8_UNORM, &pixel, vk::Sampler::null())?;

        let pipeline_cache = device.create_pipeline_cache(&[])?; //TODO: save/load cache data

        let camera = Camera::default();
        let now = Instant::now();

        let mut this = Self {
            device: device.into(),
            window_size,
            window_resized: false,
            swapchain,
            thread_cmd_pools: Default::default(),
            main_cmd_buffers: vec![],
            frame_state,
            current_frame: 0,
            default_texture,
            samplers: Default::default(),
            pipeline_cache,
            prev_frame_time: now,
            last_frame_time: now,
            camera,
            sunlight: Vec3::Y,
        };

        let cmd_pool = this.get_thread_cmd_pool()?;
        this.main_cmd_buffers =
            this.device
                .create_command_buffers(cmd_pool, MAX_FRAMES_IN_FLIGHT as u32, vk::CommandBufferLevel::PRIMARY)?;

        let sampler = this.get_sampler(vk::Filter::NEAREST, vk::Filter::NEAREST, vk::SamplerAddressMode::REPEAT)?;
        this.default_texture.sampler = sampler;

        Ok(this)
    }

    #[inline]
    pub fn instance(&self) -> &ash::Instance {
        &self.device.instance
    }

    #[inline]
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let new_size = UVec2::new(width, height);
        eprintln!("window size: {new_size}");
        if new_size != self.window_size {
            self.window_size = new_size;
            self.window_resized = true;
        }
    }

    #[inline]
    pub fn get_frame_timestamp(&self) -> Instant {
        self.last_frame_time
    }

    #[inline]
    pub fn get_frame_time(&self) -> Duration {
        self.last_frame_time - self.prev_frame_time
    }

    #[inline]
    pub fn get_frame_count(&self) -> u64 {
        self.current_frame
    }

    pub fn get_thread_cmd_pool(&self) -> VulkanResult<vk::CommandPool> {
        self.thread_cmd_pools
            .get_or_try(|| {
                self.device
                    .create_command_pool(self.device.dev_info.graphics_idx, vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            })
            .copied()
    }

    pub fn get_sampler(
        &self, mag_filter: vk::Filter, min_filter: vk::Filter, addr_mode: vk::SamplerAddressMode,
    ) -> VulkanResult<vk::Sampler> {
        use std::collections::hash_map::Entry;
        let key = SamplerOptions {
            mag_filter,
            min_filter,
            addr_mode,
        };
        match self.samplers.lock().unwrap().entry(key) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => {
                let sampler_ci = vk::SamplerCreateInfo::builder()
                    .mag_filter(mag_filter)
                    .min_filter(min_filter)
                    .address_mode_u(addr_mode)
                    .address_mode_v(addr_mode)
                    .address_mode_w(addr_mode)
                    .anisotropy_enable(false)
                    .max_anisotropy(1.0)
                    .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                    .unnormalized_coordinates(false)
                    .compare_enable(false)
                    .compare_op(vk::CompareOp::ALWAYS)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST);
                let sampler = unsafe {
                    self.device
                        .create_sampler(&sampler_ci, None)
                        .describe_err("Failed to create texture sampler")?
                };
                entry.insert(sampler);
                Ok(sampler)
            }
        }
    }

    pub fn create_texture(&self, width: u32, height: u32, data: &[u8]) -> VulkanResult<Texture> {
        let sampler = self.get_sampler(vk::Filter::LINEAR, vk::Filter::LINEAR, vk::SamplerAddressMode::REPEAT)?;
        Texture::new(&self.device, width, height, vk::Format::R8G8B8A8_SRGB, data, sampler)
    }

    fn record_primary_command_buffer(
        &self, cmd_buffer: vk::CommandBuffer, image_idx: usize, secondaries: &[vk::CommandBuffer],
    ) -> VulkanResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .describe_err("Failed to begin recording command buffer")?;
        }

        let color_attach = if let Some(msaa_imgview) = self.swapchain.msaa_imgview {
            vk::RenderingAttachmentInfo::builder()
                .image_view(msaa_imgview)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                .resolve_image_view(self.swapchain.image_views[image_idx])
                .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                })
        } else {
            vk::RenderingAttachmentInfo::builder()
                .image_view(self.swapchain.image_views[image_idx])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                })
        };
        let depth_attach = vk::RenderingAttachmentInfo::builder()
            .image_view(self.swapchain.depth_imgviews[image_idx])
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            });
        let render_info = vk::RenderingInfo::builder()
            .flags(vk::RenderingFlags::CONTENTS_SECONDARY_COMMAND_BUFFERS)
            .render_area(self.swapchain.extent_rect())
            .layer_count(1)
            .color_attachments(slice::from_ref(&color_attach))
            .depth_attachment(&depth_attach);

        unsafe {
            self.device.transition_image_layout(
                cmd_buffer,
                self.swapchain.images[image_idx],
                self.swapchain.format,
                1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
            self.device.dynrender_fn.cmd_begin_rendering(cmd_buffer, &render_info);
            self.device.cmd_execute_commands(cmd_buffer, secondaries);
            self.device.dynrender_fn.cmd_end_rendering(cmd_buffer);
            self.device.transition_image_layout(
                cmd_buffer,
                self.swapchain.images[image_idx],
                self.swapchain.format,
                1,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
            self.device
                .end_command_buffer(cmd_buffer)
                .describe_err("Failed to end recording command buffer")?;
        }
        Ok(())
    }

    pub fn begin_secondary_draw_commands(&self, cmd_buffer: vk::CommandBuffer, flags: vk::CommandBufferUsageFlags) -> VulkanResult<()> {
        let mut render_info = vk::CommandBufferInheritanceRenderingInfo::builder()
            .color_attachment_formats(slice::from_ref(&self.swapchain.format))
            .depth_attachment_format(self.swapchain.depth_format)
            .rasterization_samples(self.swapchain.samples);
        let inherit_info = vk::CommandBufferInheritanceInfo::builder().push_next(&mut render_info);
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE | flags)
            .inheritance_info(&inherit_info);
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .describe_err("Failed to begin recording secondary command buffer")?;
        }
        Ok(())
    }

    pub fn end_secondary_draw_commands(&self, cmd_buffer: vk::CommandBuffer) -> VulkanResult<vk::CommandBuffer> {
        unsafe {
            self.device
                .end_command_buffer(cmd_buffer)
                .describe_err("Failed to end recording secodary command buffer")?;
        }
        Ok(cmd_buffer)
    }

    pub fn submit_draw_commands(&mut self, draw_commands: impl IntoIterator<Item = DrawPayload>) -> VulkanResult<bool> {
        let frame_idx = (self.current_frame % self.frame_state.len() as u64) as usize;
        let frame = &mut self.frame_state[frame_idx];
        let image_avail_sem = frame.image_avail_sem;
        let render_finish_sem = frame.render_finished_sem;
        let in_flight_sem = frame.in_flight_sem;
        let command_buffer = self.main_cmd_buffers[frame_idx];

        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(slice::from_ref(&in_flight_sem))
            .values(slice::from_ref(&frame.wait_frame));
        unsafe {
            self.device
                .wait_semaphores(&wait_info, u64::MAX)
                .describe_err("Failed waiting semaphore")?;
        }
        // all previous work for this frame is done at this point
        frame.free_payload(&self.device)?;
        self.prev_frame_time = self.last_frame_time;
        self.last_frame_time = Instant::now();

        let image_idx = unsafe {
            let acquire_res = self
                .device
                .swapchain_fn
                .acquire_next_image(*self.swapchain, u64::MAX, image_avail_sem, vk::Fence::null());
            match acquire_res {
                Ok((idx, _)) => idx,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    eprintln!("swapchain out of date");
                    self.swapchain.recreate(&self.device, self.window_size)?;
                    return Ok(false);
                }
                Err(e) => return Err(VkError::VulkanMsg("Failed to acquire swapchain image", e)),
            }
        };

        unsafe {
            self.device
                .reset_command_buffer(command_buffer, Default::default())
                .describe_err("Failed to reset command buffer")?;
        }

        let next_frame = self.current_frame + 1;
        frame.wait_frame = next_frame;
        frame.payload.extend(draw_commands);
        let cmd_buffers: Vec<_> = frame.payload.iter().map(|pl| pl.cmd_buffer).collect();
        self.record_primary_command_buffer(command_buffer, image_idx as _, &cmd_buffers)?;

        let signal_values = [0, next_frame];
        let signal_sems = [render_finish_sem, in_flight_sem];
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::builder().signal_semaphore_values(&signal_values);
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(slice::from_ref(&image_avail_sem))
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(slice::from_ref(&command_buffer))
            .signal_semaphores(&signal_sems)
            .push_next(&mut timeline_info);
        unsafe {
            self.device
                .queue_submit(self.device.graphics_queue, slice::from_ref(&submit_info), vk::Fence::null())
                .describe_err("Failed to submit draw command buffer")?
        }

        self.current_frame += 1;

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

        if suboptimal || self.window_resized {
            eprintln!("swapchain suboptimal");
            self.swapchain.recreate(&self.device, self.window_size)?;
            self.window_resized = false;
        }

        Ok(true)
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.frame_state.cleanup(&self.device);
            self.thread_cmd_pools.cleanup(&self.device);
            self.swapchain.cleanup(&self.device);
            self.samplers.lock().unwrap().cleanup(&self.device);
            self.default_texture.cleanup(&self.device);
            self.device.destroy_pipeline_cache(self.pipeline_cache, None);
        }
    }
}

#[derive(Debug, Clone)]
pub struct PipelineBuilder<'a> {
    pub shader: &'a Shader,
    pub desc_layouts: Vec<vk::DescriptorSetLayout>,
    pub push_constants: &'a [vk::PushConstantRange],
    pub binding_desc: Vec<vk::VertexInputBindingDescription>,
    pub attrib_desc: Vec<vk::VertexInputAttributeDescription>,
    pub color_format: vk::Format,
    pub depth_format: vk::Format,
    pub samples: vk::SampleCountFlags,
    pub mode: PipelineMode,
    pub topology: vk::PrimitiveTopology,
}

impl<'a> PipelineBuilder<'a> {
    pub fn new(shader: &'a Shader) -> Self {
        Self {
            shader,
            desc_layouts: vec![],
            push_constants: &[],
            binding_desc: vec![],
            attrib_desc: vec![],
            color_format: vk::Format::UNDEFINED,
            depth_format: vk::Format::UNDEFINED,
            samples: vk::SampleCountFlags::TYPE_1,
            mode: PipelineMode::Opaque,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        }
    }

    pub fn vertex_input<V: VertexInput>(mut self) -> Self {
        self.binding_desc = vec![V::binding_desc(0)];
        self.attrib_desc = V::attr_desc(0);
        self
    }

    pub fn descriptor_layout(mut self, set_layout: vk::DescriptorSetLayout) -> Self {
        self.desc_layouts = vec![set_layout];
        self
    }

    pub fn push_constants(mut self, push_constants: &'a [vk::PushConstantRange]) -> Self {
        self.push_constants = push_constants;
        self
    }

    pub fn render_to_swapchain(mut self, swapchain: &Swapchain) -> Self {
        self.color_format = swapchain.format;
        self.depth_format = swapchain.depth_format;
        self.samples = swapchain.samples;
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

    pub fn build(self, engine: &VulkanEngine) -> VulkanResult<Pipeline> {
        let layout = engine.device.create_pipeline_layout(&self.desc_layouts, self.push_constants)?;
        let handle = Pipeline::create_pipeline(&engine.device, layout, self, engine.pipeline_cache)?;
        Ok(Pipeline { handle, layout })
    }
}

#[derive(Debug)]
pub struct Pipeline {
    pub(crate) handle: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
}

impl Pipeline {
    #[inline]
    pub fn builder(shader: &Shader) -> PipelineBuilder {
        PipelineBuilder::new(shader)
    }

    fn create_pipeline(
        device: &ash::Device, layout: vk::PipelineLayout, params: PipelineBuilder, cache: vk::PipelineCache,
    ) -> VulkanResult<vk::Pipeline> {
        let entry_point = cstr!("main");
        let shader_stages_ci = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(params.shader.vert)
                .name(entry_point)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(params.shader.frag)
                .name(entry_point)
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
            .front_face(vk::FrontFace::CLOCKWISE);

        let multisample_ci = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(params.samples)
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
                .create_graphics_pipelines(cache, slice::from_ref(&pipeline_ci), None)
                .map_err(|(_, err)| VkError::VulkanMsg("Error creating pipeline", err))?
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

#[derive(Debug, Clone, Copy)]
pub enum PipelineMode {
    Opaque,
    Transparent,
    Background,
    Overlay,
}

impl PipelineMode {
    fn depth_test(self) -> bool {
        match self {
            Self::Overlay => false,
            _ => true,
        }
    }

    fn depth_write(self) -> bool {
        match self {
            Self::Opaque => true,
            _ => false,
        }
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
            Self::Overlay => vk::CullModeFlags::NONE,
            _ => vk::CullModeFlags::BACK,
        }
    }

    fn blend_enable(self) -> bool {
        match self {
            Self::Transparent | Self::Overlay => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CmdbufAction {
    None,
    Reset,
    Free(vk::CommandPool),
}

pub struct DrawPayload {
    pub cmd_buffer: vk::CommandBuffer,
    pub cmdbuf_action: CmdbufAction,
    pub on_frame_finish: Option<Box<dyn FnOnce(&VulkanDevice) + Send + Sync>>,
}

impl DrawPayload {
    #[inline]
    pub fn new(cmd_buffer: vk::CommandBuffer, cmdbuf_action: CmdbufAction) -> Self {
        Self {
            cmd_buffer,
            cmdbuf_action,
            on_frame_finish: None,
        }
    }

    #[inline]
    pub fn new_with_callback<F>(cmd_buffer: vk::CommandBuffer, cmdbuf_action: CmdbufAction, on_frame_finish: F) -> Self
    where
        F: FnOnce(&VulkanDevice) + Send + Sync + 'static,
    {
        Self {
            cmd_buffer,
            cmdbuf_action,
            on_frame_finish: Some(Box::new(on_frame_finish)),
        }
    }
}

struct FrameState {
    image_avail_sem: vk::Semaphore,     // present -> image acquired
    render_finished_sem: vk::Semaphore, // queue submit -> present
    in_flight_sem: vk::Semaphore,       // queue submit -> frame finished
    wait_frame: u64,
    payload: Vec<DrawPayload>,
}

impl FrameState {
    fn new(device: &VulkanDevice) -> VulkanResult<Self> {
        let image_avail_sem = device.create_semaphore(vk::SemaphoreType::BINARY)?;
        let render_finished_sem = device.create_semaphore(vk::SemaphoreType::BINARY)?;
        let in_flight_sem = device.create_semaphore(vk::SemaphoreType::TIMELINE)?;
        device.debug(|d| {
            d.set_object_name(device, &image_avail_sem, "Image available semaphore");
            d.set_object_name(device, &render_finished_sem, "Render finished semaphore");
            d.set_object_name(device, &in_flight_sem, "In-flight semaphore")
        });
        Ok(Self {
            image_avail_sem,
            render_finished_sem,
            in_flight_sem,
            wait_frame: 0,
            payload: vec![],
        })
    }

    fn free_payload(&mut self, device: &VulkanDevice) -> VulkanResult<()> {
        for pl in self.payload.drain(..) {
            match pl.cmdbuf_action {
                CmdbufAction::None => (),
                CmdbufAction::Reset => unsafe {
                    device.reset_command_buffer(pl.cmd_buffer, Default::default())?;
                },
                CmdbufAction::Free(pool) => unsafe {
                    device.free_command_buffers(pool, &[pl.cmd_buffer]);
                },
            }
            if let Some(f) = pl.on_frame_finish {
                f(device)
            }
        }
        Ok(())
    }
}

impl Cleanup<VulkanDevice> for FrameState {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_semaphore(self.image_avail_sem, None);
        device.destroy_semaphore(self.render_finished_sem, None);
        device.destroy_semaphore(self.in_flight_sem, None);
    }
}

#[derive(Debug)]
pub struct Texture {
    image: VkImage,
    imgview: vk::ImageView,
    sampler: vk::Sampler,
}

impl Texture {
    pub fn new(
        device: &VulkanDevice, width: u32, height: u32, format: vk::Format, data: &[u8], sampler: vk::Sampler,
    ) -> VulkanResult<Self> {
        let image = device.create_image_from_data(width, height, format, ImageData::Single(data), Default::default())?;
        let imgview = device.create_image_view(*image, format, 0, vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR)?;
        Ok(Self { image, imgview, sampler })
    }

    pub fn new_cubemap(
        device: &VulkanDevice, width: u32, height: u32, format: vk::Format, data: &[&[u8]; 6], sampler: vk::Sampler,
    ) -> VulkanResult<Self> {
        let image = device.create_image_from_data(width, height, format, ImageData::Array(data), vk::ImageCreateFlags::CUBE_COMPATIBLE)?;
        let imgview = device.create_image_view(*image, format, 0, vk::ImageViewType::CUBE, vk::ImageAspectFlags::COLOR)?;
        Ok(Self { image, imgview, sampler })
    }

    pub fn update(&mut self, device: &VulkanDevice, x: u32, y: u32, width: u32, height: u32, data: &[u8]) -> VulkanResult<()> {
        //TODO: validate params
        device.update_image_from_data(*self.image, x as _, y as _, width, height, ImageData::Single(data))
    }

    pub(crate) fn descriptor(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image_view: self.imgview,
            sampler: self.sampler,
        }
    }
}

impl Cleanup<VulkanDevice> for Texture {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        self.imgview.cleanup(device);
        self.image.cleanup(device);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SamplerOptions {
    mag_filter: vk::Filter,
    min_filter: vk::Filter,
    addr_mode: vk::SamplerAddressMode,
}

#[derive(Debug)]
pub struct Shader {
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
}

impl Shader {
    pub fn new(device: &VulkanDevice, vert_spv: &[u32], frag_spv: &[u32]) -> VulkanResult<Self> {
        let vert = device.create_shader_module(vert_spv)?;
        let frag = device.create_shader_module(frag_spv)?;
        Ok(Self { vert, frag })
    }
}

impl Cleanup<VulkanDevice> for Shader {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_shader_module(self.vert, None);
        device.destroy_shader_module(self.frag, None);
    }
}
