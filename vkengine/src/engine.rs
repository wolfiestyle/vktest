use crate::camera::Camera;
use crate::create::CreateFromInfo;
use crate::device::{ImageData, ImageParams, MappedMemory, VkBuffer, VkImage, VulkanDevice};
use crate::instance::DeviceSelection;
use crate::swapchain::Swapchain;
use crate::types::*;
use ash::vk;
use glam::{Mat4, UVec2, Vec3};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::collections::HashMap;
use std::slice;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const SWAPCHAIN_IMAGE_COUNT: u32 = 3;
pub const QUEUE_DEPTH: usize = 2;

#[derive(Debug)]
pub struct VulkanEngine {
    pub(crate) device: Arc<VulkanDevice>,
    window_size: UVec2,
    window_resized: bool,
    pub(crate) swapchain: Swapchain,
    main_cmd_buffers: CmdBufferRing,
    frame_state: Vec<FrameState>,
    current_frame: u64,
    samplers: Mutex<HashMap<SamplerOptions, vk::Sampler>>,
    pub(crate) default_texture: Texture,
    pub(crate) pipeline_cache: vk::PipelineCache,
    prev_frame_time: Instant,
    last_frame_time: Instant,
    gpu_time: u64,
    pub camera: Camera,
    pub projection: Mat4,
    pub view_proj: Mat4,
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

        let main_cmd_buffers = CmdBufferRing::new_with_level(&device, vk::CommandBufferLevel::PRIMARY)?;
        let frame_state = (0..QUEUE_DEPTH).map(|_| FrameState::new(&device)).collect::<Result<Vec<_>, _>>()?;

        let default_texture = Texture::new(
            &device,
            1,
            1,
            vk::Format::R8G8B8A8_UNORM,
            ImageData::Single(&[255; 4]),
            vk::Sampler::null(),
            false,
        )?;

        let pipeline_cache = vk::PipelineCacheCreateInfo::builder().create(&device)?; //TODO: save/load cache data

        let camera = Camera::default();
        let now = Instant::now();

        let mut this = Self {
            device: device.into(),
            window_size,
            window_resized: false,
            swapchain,
            main_cmd_buffers,
            frame_state,
            current_frame: 0,
            default_texture,
            samplers: Default::default(),
            pipeline_cache,
            prev_frame_time: now,
            last_frame_time: now,
            gpu_time: 0,
            camera,
            projection: Mat4::IDENTITY,
            view_proj: Mat4::IDENTITY,
            sunlight: Vec3::NEG_Y,
        };

        let sampler = this.get_sampler(
            vk::Filter::NEAREST,
            vk::Filter::NEAREST,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::REPEAT,
            false,
        )?;
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
    pub fn get_msaa_samples(&self) -> u32 {
        self.swapchain.samples.as_raw()
    }

    pub fn set_msaa_samples(&mut self, samples: u32) -> VulkanResult<()> {
        if !samples.is_power_of_two() {
            return VkError::InvalidArgument("Sample count is not power of two").into();
        }
        let samples = vk::SampleCountFlags::from_raw(samples);
        if !self.device.dev_info.msaa_support.contains(samples) {
            return VkError::InvalidArgument("Unsupported sample count").into();
        }
        if samples != self.swapchain.samples {
            self.swapchain.samples = samples;
            self.swapchain.recreate(&self.device, self.window_size)?;
        }
        Ok(())
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
    pub fn get_current_frame(&self) -> u64 {
        self.current_frame
    }

    pub fn get_gpu_time(&self) -> Duration {
        Duration::from_nanos((self.gpu_time as f64 * self.device.dev_info.timestamp_period as f64) as u64)
    }

    pub fn get_sampler(
        &self, mag_filter: vk::Filter, min_filter: vk::Filter, wrap_u: vk::SamplerAddressMode, wrap_v: vk::SamplerAddressMode,
        aniso_enabled: bool,
    ) -> VulkanResult<vk::Sampler> {
        use std::collections::hash_map::Entry;
        let key = SamplerOptions {
            mag_filter,
            min_filter,
            wrap_u,
            wrap_v,
            aniso_enabled,
        };
        match self.samplers.lock().unwrap().entry(key) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => {
                let sampler = vk::SamplerCreateInfo::builder()
                    .mag_filter(mag_filter)
                    .min_filter(min_filter)
                    .address_mode_u(wrap_u)
                    .address_mode_v(wrap_v)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .anisotropy_enable(aniso_enabled)
                    .max_anisotropy(self.device.dev_info.max_aniso)
                    .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                    .unnormalized_coordinates(false)
                    .compare_enable(false)
                    .compare_op(vk::CompareOp::ALWAYS)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .create(&self.device)?;
                entry.insert(sampler);
                Ok(sampler)
            }
        }
    }

    pub fn create_texture(&self, tex_data: &gltf_import::Texture, gltf: &gltf_import::Gltf) -> VulkanResult<Texture> {
        use gltf_import::{MagFilter, MinFilter, WrappingMode};

        let image_info = &gltf[tex_data.image];
        let gltf_import::ImageData::Decoded(image) = &image_info.data else { return Err(VkError::EngineError("missing texture image")) };

        let mag = match tex_data.mag_filter {
            MagFilter::Nearest => vk::Filter::NEAREST,
            MagFilter::Linear => vk::Filter::LINEAR,
        };
        let min = match tex_data.min_filter {
            MinFilter::Nearest | MinFilter::NearestMipmapNearest | MinFilter::NearestMipmapLinear => vk::Filter::NEAREST,
            MinFilter::Linear | MinFilter::LinearMipmapNearest | MinFilter::LinearMipmapLinear => vk::Filter::LINEAR,
        };
        let wrap_u = match tex_data.wrap_u {
            WrappingMode::Repeat => vk::SamplerAddressMode::REPEAT,
            WrappingMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            WrappingMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        };
        let wrap_v = match tex_data.wrap_v {
            WrappingMode::Repeat => vk::SamplerAddressMode::REPEAT,
            WrappingMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            WrappingMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        };
        let sampler = self.get_sampler(mag, min, wrap_u, wrap_v, true)?;

        let format = if image_info.srgb {
            vk::Format::R8G8B8A8_SRGB
        } else {
            vk::Format::R8G8B8A8_UNORM
        };
        Texture::new(
            &self.device,
            image.width(),
            image.height(),
            format,
            ImageData::Single(image.to_rgba8().as_raw()),
            sampler,
            true,
        )
    }

    fn record_primary_command_buffer(
        &self, cmd_buffer: vk::CommandBuffer, draw_cmds: &[DrawPayload], image_idx: usize, frame: &FrameState,
    ) -> VulkanResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .describe_err("Failed to begin recording command buffer")?;
            self.device.cmd_reset_query_pool(cmd_buffer, frame.time_query, 0, 2);
            self.device
                .cmd_write_timestamp(cmd_buffer, vk::PipelineStageFlags::TOP_OF_PIPE, frame.time_query, 0);
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
            .image_view(self.swapchain.depth_imgview.expect("missing depth image view"))
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

        self.device.transition_image_layout(
            cmd_buffer,
            self.swapchain.images[image_idx],
            Swapchain::SUBRESOURCE_RANGE,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        self.device.image_reuse_barrier(
            cmd_buffer,
            self.swapchain.depth_image.as_ref().expect("missing depth image").handle,
            self.swapchain.depth_format,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );
        if let Some(image) = &self.swapchain.msaa_image {
            self.device.image_reuse_barrier(
                cmd_buffer,
                image.handle,
                self.swapchain.format,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        }
        unsafe {
            self.device.dynrender_fn.cmd_begin_rendering(cmd_buffer, &render_info);
            for cmdbuf in draw_cmds {
                self.device.cmd_execute_commands(cmd_buffer, slice::from_ref(&cmdbuf.cmd_buffer));
            }
            self.device.dynrender_fn.cmd_end_rendering(cmd_buffer);
            self.device.transition_image_layout(
                cmd_buffer,
                self.swapchain.images[image_idx],
                Swapchain::SUBRESOURCE_RANGE,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
            self.device
                .cmd_write_timestamp(cmd_buffer, vk::PipelineStageFlags::BOTTOM_OF_PIPE, frame.time_query, 1);
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
                .describe_err("Failed to end recording secondary command buffer")?;
        }
        Ok(cmd_buffer)
    }

    pub fn update(&mut self) {
        let view = self.camera.get_view_transform();
        self.projection = self.camera.get_projection(self.swapchain.aspect());
        self.view_proj = self.projection * view;
    }

    pub fn submit_draw_commands(&mut self, draw_commands: impl IntoIterator<Item = DrawPayload>) -> VulkanResult<bool> {
        let frame_idx = (self.current_frame % self.frame_state.len() as u64) as usize;
        let frame = &self.frame_state[frame_idx];
        let image_avail_sem = frame.image_avail_sem;
        let render_finish_sem = frame.render_finished_sem;
        let in_flight_sem = frame.in_flight_sem;
        let command_buffer = self.main_cmd_buffers[self.current_frame];

        // the swapchain image id is available before it's actually ready to use
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

        // ..that's all the info we need to record a new command buffer
        unsafe {
            self.device
                .reset_command_buffer(command_buffer, Default::default())
                .describe_err("Failed to reset command buffer")?;
        }
        let draw_cmds: Vec<_> = draw_commands.into_iter().collect();
        self.record_primary_command_buffer(command_buffer, &draw_cmds, image_idx as _, frame)?;

        // now we can wait for frame finished, then we can modify resources
        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(slice::from_ref(&in_flight_sem))
            .values(slice::from_ref(&frame.wait_frame));
        unsafe {
            self.device
                .wait_semaphores(&wait_info, u64::MAX)
                .describe_err("Failed waiting semaphore")?;
        }
        let frame = &mut self.frame_state[frame_idx];
        frame.execute_on_finish(&self.device);
        let next_frame = self.current_frame + 1;
        frame.wait_frame = next_frame;
        frame.payload.extend(draw_cmds);
        // get timestamp data
        let mut query_data = [0u64; 2];
        let query_res = unsafe {
            self.device
                .get_query_pool_results(frame.time_query, 0, 2, &mut query_data, vk::QueryResultFlags::TYPE_64)
        };
        if query_res.is_ok() {
            self.gpu_time = query_data[1].saturating_sub(query_data[0]);
        }
        self.prev_frame_time = self.last_frame_time;
        self.last_frame_time = Instant::now();

        // submit work for the next frame
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
            self.main_cmd_buffers.cleanup(&self.device);
            self.swapchain.cleanup(&self.device);
            self.samplers.lock().unwrap().cleanup(&self.device);
            self.default_texture.cleanup(&self.device);
            self.device.destroy_pipeline_cache(self.pipeline_cache, None);
        }
    }
}

#[derive(Debug)]
pub struct CmdBufferRing {
    pool: vk::CommandPool,
    buffers: [vk::CommandBuffer; QUEUE_DEPTH + 1],
}

impl CmdBufferRing {
    #[inline]
    pub fn new(device: &VulkanDevice) -> VulkanResult<Self> {
        Self::new_with_level(device, vk::CommandBufferLevel::SECONDARY)
    }

    fn new_with_level(device: &VulkanDevice, level: vk::CommandBufferLevel) -> VulkanResult<Self> {
        let pool = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(device.dev_info.graphics_idx)
            .create(device)?;
        // we use N + 1 command buffers so we can write on them right away without waiting for the previous frame
        let buffers = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(level)
            .command_buffer_count(QUEUE_DEPTH as u32 + 1)
            .create(device)?
            .try_into()
            .unwrap();
        Ok(Self { pool, buffers })
    }

    pub fn get_current_buffer(&self, engine: &VulkanEngine) -> VulkanResult<vk::CommandBuffer> {
        let cmd_buffer = self[engine.current_frame];
        unsafe { engine.device.reset_command_buffer(cmd_buffer, Default::default())? };
        Ok(cmd_buffer)
    }
}

impl std::ops::Index<u64> for CmdBufferRing {
    type Output = vk::CommandBuffer;

    #[inline]
    fn index(&self, index: u64) -> &Self::Output {
        &self.buffers[(index % (QUEUE_DEPTH as u64 + 1)) as usize]
    }
}

impl Cleanup<VulkanDevice> for CmdBufferRing {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        self.pool.cleanup(device);
    }
}

#[derive(Debug)]
pub struct UploadBuffer {
    buffers: [VkBuffer; QUEUE_DEPTH + 1],
}

impl UploadBuffer {
    pub fn new(device: &VulkanDevice, size: vk::DeviceSize, usage: vk::BufferUsageFlags, name: &str) -> VulkanResult<Self> {
        let buffers = (0..QUEUE_DEPTH + 1)
            .map(|_| device.allocate_cpu_buffer(size, usage, name))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .unwrap();
        let this = Self { buffers };
        device.debug(|d| this.buffers.iter().for_each(|b| d.set_object_name(device, &b.handle, name)));
        Ok(this)
    }

    #[inline]
    pub fn get_current_buffer(&self, engine: &VulkanEngine) -> &VkBuffer {
        &self[engine.current_frame]
    }

    #[inline]
    pub fn get_current_buffer_mut(&mut self, engine: &VulkanEngine) -> &mut VkBuffer {
        &mut self[engine.current_frame]
    }

    pub fn map(&mut self, engine: &VulkanEngine) -> VulkanResult<MappedMemory> {
        self.get_current_buffer_mut(engine).map()
    }

    pub fn descriptor(&self, engine: &VulkanEngine) -> vk::DescriptorBufferInfo {
        self.get_current_buffer(engine).descriptor()
    }

    pub fn descriptor_slice(&self, offset: u64, range: u64, engine: &VulkanEngine) -> vk::DescriptorBufferInfo {
        self.get_current_buffer(engine).descriptor_slice(offset, range)
    }
}

impl std::ops::Index<u64> for UploadBuffer {
    type Output = VkBuffer;

    #[inline]
    fn index(&self, index: u64) -> &Self::Output {
        &self.buffers[(index % (QUEUE_DEPTH as u64 + 1)) as usize]
    }
}

impl std::ops::IndexMut<u64> for UploadBuffer {
    #[inline]
    fn index_mut(&mut self, index: u64) -> &mut Self::Output {
        &mut self.buffers[(index % (QUEUE_DEPTH as u64 + 1)) as usize]
    }
}

impl Cleanup<VulkanDevice> for UploadBuffer {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        self.buffers.cleanup(device)
    }
}

pub struct DrawPayload {
    pub cmd_buffer: vk::CommandBuffer,
    pub on_frame_finish: Option<Box<dyn FnOnce(&VulkanDevice) + Send + Sync>>,
}

impl std::fmt::Debug for DrawPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrawPayload")
            .field("cmd_buffer", &self.cmd_buffer)
            .field("on_frame_finish", &self.on_frame_finish.as_ref().map(|_| "..."))
            .finish()
    }
}

impl DrawPayload {
    #[inline]
    pub fn new(cmd_buffer: vk::CommandBuffer) -> Self {
        Self {
            cmd_buffer,
            on_frame_finish: None,
        }
    }

    #[inline]
    pub fn new_with_callback<F>(cmd_buffer: vk::CommandBuffer, on_frame_finish: F) -> Self
    where
        F: FnOnce(&VulkanDevice) + Send + Sync + 'static,
    {
        Self {
            cmd_buffer,
            on_frame_finish: Some(Box::new(on_frame_finish)),
        }
    }
}

#[derive(Debug)]
struct FrameState {
    image_avail_sem: vk::Semaphore,     // present -> image acquired
    render_finished_sem: vk::Semaphore, // queue submit -> present
    in_flight_sem: vk::Semaphore,       // queue submit -> frame finished
    wait_frame: u64,
    payload: Vec<DrawPayload>,
    time_query: vk::QueryPool,
}

impl FrameState {
    fn new(device: &VulkanDevice) -> VulkanResult<Self> {
        let image_avail_sem = device.make_semaphore(vk::SemaphoreType::BINARY)?;
        let render_finished_sem = device.make_semaphore(vk::SemaphoreType::BINARY)?;
        let in_flight_sem = device.make_semaphore(vk::SemaphoreType::TIMELINE)?;
        let time_query = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2)
            .create(device)?;
        device.debug(|d| {
            d.set_object_name(device, &image_avail_sem, "Image available semaphore");
            d.set_object_name(device, &render_finished_sem, "Render finished semaphore");
            d.set_object_name(device, &in_flight_sem, "In-flight semaphore");
            d.set_object_name(device, &time_query, "Timestamp QueryPool");
        });
        Ok(Self {
            image_avail_sem,
            render_finished_sem,
            in_flight_sem,
            wait_frame: 0,
            payload: vec![],
            time_query,
        })
    }

    fn execute_on_finish(&mut self, device: &VulkanDevice) {
        for pl in self.payload.drain(..) {
            if let Some(f) = pl.on_frame_finish {
                f(device)
            }
        }
    }
}

impl Cleanup<VulkanDevice> for FrameState {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_semaphore(self.image_avail_sem, None);
        device.destroy_semaphore(self.render_finished_sem, None);
        device.destroy_semaphore(self.in_flight_sem, None);
        device.destroy_query_pool(self.time_query, None);
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
        device: &VulkanDevice, width: u32, height: u32, format: vk::Format, data: ImageData, sampler: vk::Sampler, gen_mipmaps: bool,
    ) -> VulkanResult<Self> {
        let params = ImageParams {
            width,
            height,
            format,
            layers: data.layer_count(),
            mip_levels: if gen_mipmaps { width.max(height).ilog2() + 1 } else { 1 },
            ..Default::default()
        };
        let image = device.create_image_from_data(params, data, data.image_create_flags())?;
        let imgview = vk::ImageViewCreateInfo::builder()
            .image(*image)
            .format(format)
            .view_type(data.view_type())
            .subresource_range(image.props.subresource_range())
            .create(device)?;
        Ok(Self { image, imgview, sampler })
    }

    pub fn update(&mut self, device: &VulkanDevice, x: u32, y: u32, width: u32, height: u32, data: &[u8]) -> VulkanResult<()> {
        let w = self.image.props.width;
        let h = self.image.props.height;
        if x >= w || y >= h || x + width >= w || y + height >= h {
            return VkError::InvalidArgument("Texture update rect out of bounds").into();
        }
        device.update_image_from_data(&self.image, x as _, y as _, width, height, 0, ImageData::Single(data))
    }

    pub fn descriptor(&self) -> vk::DescriptorImageInfo {
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
    wrap_u: vk::SamplerAddressMode,
    wrap_v: vk::SamplerAddressMode,
    aniso_enabled: bool,
}

#[derive(Debug)]
pub struct Shader {
    pub vert: vk::ShaderModule,
    pub frag: vk::ShaderModule,
}

impl Shader {
    pub fn new(device: &VulkanDevice, vert_spv: &[u32], frag_spv: &[u32]) -> VulkanResult<Self> {
        let vert = vk::ShaderModuleCreateInfo::builder().code(vert_spv).create(device)?;
        let frag = vk::ShaderModuleCreateInfo::builder().code(frag_spv).create(device)?;
        Ok(Self { vert, frag })
    }
}

impl Cleanup<VulkanDevice> for Shader {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.destroy_shader_module(self.vert, None);
        device.destroy_shader_module(self.frag, None);
    }
}
