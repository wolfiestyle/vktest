use crate::vulkan::device::{SwapchainInfo, VulkanDevice};
use crate::vulkan::types::*;
use ash::vk;
use inline_spirv::include_spirv;
use winit::window::Window;

const SWAPCHAIN_IMAGE_COUNT: u32 = 3;
const MAX_FRAMES_IN_FLIGHT: usize = 2;
pub type Vertex = ([f32; 2], [f32; 3]);

pub struct VulkanApp {
    device: VulkanDevice,
    swapchain: SwapchainInfo,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    transfer_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    sync: Vec<FrameSyncState>,
    current_frame: usize,
    vertex_buffer: vk::Buffer,
    vb_memory: vk::DeviceMemory,
}

impl VulkanApp {
    pub fn new(window: &Window) -> VulkanResult<Self> {
        let vk = VulkanDevice::new(window)?;
        let swapchain = vk.create_swapchain(window, SWAPCHAIN_IMAGE_COUNT, None)?;

        let vert_spv = include_spirv!("src/shaders/color.vert.glsl", vert, glsl);
        let frag_spv = include_spirv!("src/shaders/color.frag.glsl", frag, glsl);
        let render_pass = vk.create_render_pass(swapchain.format)?;
        let framebuffers = vk.create_framebuffers(&swapchain, render_pass)?;
        let (pipeline, pipeline_layout) = vk.create_graphics_pipeline(vert_spv, frag_spv, render_pass)?;

        let command_pool = vk.create_command_pool(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)?;
        let transfer_pool = vk.create_command_pool(vk::CommandPoolCreateFlags::TRANSIENT)?;
        let command_buffers = vk.create_command_buffers(command_pool, MAX_FRAMES_IN_FLIGHT as u32)?;
        let sync = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| FrameSyncState::new(&vk))
            .collect::<Result<_, _>>()?;

        let vertices: [Vertex; 3] = [
            ([0.0, -0.5], [1.0, 0.0, 0.0]),
            ([0.5, 0.5], [0.0, 1.0, 0.0]),
            ([-0.5, 0.5f32], [0.0, 0.0, 1.0f32]),
        ];
        let (vertex_buffer, vb_memory) = vk.create_buffer(&vertices, vk::BufferUsageFlags::VERTEX_BUFFER, transfer_pool)?;

        Ok(Self {
            device: vk,
            swapchain,
            render_pass,
            pipeline,
            pipeline_layout,
            framebuffers,
            command_pool,
            transfer_pool,
            command_buffers,
            sync,
            current_frame: 0,
            vertex_buffer,
            vb_memory,
        })
    }

    fn record_command_buffer(&self, cmd_buffer: vk::CommandBuffer, image_idx: u32) -> VulkanResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .describe_err("Failed to begin recording command buffer")?;
        }

        let clear_color = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let renderpass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_idx as usize])
            .render_area(self.swapchain.extent_rect())
            .clear_values(&clear_color);

        let viewport = [self.swapchain.viewport()];
        let scissor = [self.swapchain.extent_rect()];
        let buffers = [self.vertex_buffer];
        let offsets = [0];
        unsafe {
            self.device
                .cmd_begin_render_pass(cmd_buffer, &renderpass_info, vk::SubpassContents::INLINE);
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_vertex_buffers(cmd_buffer, 0, &buffers, &offsets);
            self.device.cmd_set_viewport(cmd_buffer, 0, &viewport);
            self.device.cmd_set_scissor(cmd_buffer, 0, &scissor);
            self.device.cmd_draw(cmd_buffer, 3, 1, 0, 0);
            self.device.cmd_end_render_pass(cmd_buffer);
            self.device
                .end_command_buffer(cmd_buffer)
                .describe_err("Failed to end recording command buffer")?;
        }

        Ok(())
    }

    pub fn draw_frame(&mut self, window: &Window) -> VulkanResult<()> {
        let in_flight_fen = [self.sync[self.current_frame].in_flight_fen];
        let image_avail_sem = [self.sync[self.current_frame].image_avail_sem];
        let render_finish_sem = [self.sync[self.current_frame].render_finished_sem];
        let command_buffer = [self.command_buffers[self.current_frame]];

        let image_idx = unsafe {
            self.device
                .wait_for_fences(&in_flight_fen, true, u64::MAX)
                .describe_err("Failed waiting for error")?;
            let acquire_res =
                self.device
                    .swapchain_utils
                    .acquire_next_image(self.swapchain.handle, u64::MAX, image_avail_sem[0], vk::Fence::null());
            match acquire_res {
                Ok((idx, _)) => idx,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    eprintln!("swapchain out of date");
                    self.recreate_swapchain(window)?;
                    return Ok(());
                }
                Err(e) => return Err(VkError::VulkanMsg("Failed to acquire swapchain image", e)),
            }
        };

        unsafe {
            self.device.reset_fences(&in_flight_fen).describe_err("Failed resetting fences")?;
            self.device
                .reset_command_buffer(command_buffer[0], vk::CommandBufferResetFlags::empty())
                .describe_err("Failed to reset command buffer")?;
        }

        self.record_command_buffer(command_buffer[0], image_idx)?;

        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let submit_info = [vk::SubmitInfo::builder()
            .wait_semaphores(&image_avail_sem)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffer)
            .signal_semaphores(&render_finish_sem)
            .build()];

        unsafe {
            self.device
                .queue_submit(self.device.graphics_queue, &submit_info, in_flight_fen[0])
                .describe_err("Failed to submit draw command buffer")?
        }

        let swapchains = [self.swapchain.handle];
        let image_indices = [image_idx];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&render_finish_sem)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let suboptimal = unsafe {
            self.device
                .swapchain_utils
                .queue_present(self.device.present_queue, &present_info)
                .describe_err("Failed to present queue")?
        };

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        if suboptimal {
            eprintln!("swapchain suboptimal");
            self.recreate_swapchain(window)?;
        }

        Ok(())
    }

    unsafe fn cleanup_swapchain(&mut self) {
        for &fb in &self.framebuffers {
            self.device.destroy_framebuffer(fb, None);
        }
        self.swapchain.cleanup(&self.device);
    }

    fn recreate_swapchain(&mut self, window: &Window) -> VulkanResult<()> {
        self.device.wait_idle()?;
        self.device.update_surface_info()?;
        let swapchain = self
            .device
            .create_swapchain(window, SWAPCHAIN_IMAGE_COUNT, Some(self.swapchain.handle))?;
        let framebuffers = self.device.create_framebuffers(&swapchain, self.render_pass)?;
        unsafe { self.cleanup_swapchain() };
        self.swapchain = swapchain;
        self.framebuffers = framebuffers;

        Ok(())
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.device.wait_idle().unwrap();
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vb_memory, None);
            for elem in &mut self.sync {
                elem.cleanup(&self.device);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_command_pool(self.transfer_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.cleanup_swapchain();
        }
    }
}

struct FrameSyncState {
    image_avail_sem: vk::Semaphore,
    render_finished_sem: vk::Semaphore,
    in_flight_fen: vk::Fence,
}

impl FrameSyncState {
    fn new(device: &VulkanDevice) -> VulkanResult<Self> {
        Ok(Self {
            image_avail_sem: device.create_semaphore()?,
            render_finished_sem: device.create_semaphore()?,
            in_flight_fen: device.create_fence()?,
        })
    }

    unsafe fn cleanup(&mut self, device: &ash::Device) {
        device.destroy_semaphore(self.image_avail_sem, None);
        device.destroy_semaphore(self.render_finished_sem, None);
        device.destroy_fence(self.in_flight_fen, None);
    }
}
