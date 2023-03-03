use crate::device::{VkBuffer, VulkanDevice};
use crate::engine::{DrawPayload, Pipeline, PipelineMode, Shader, Texture, VulkanEngine};
use crate::types::{Cleanup, ErrorDescription, VulkanResult};
use ash::vk;
use egui::epaint::{Primitive, Vertex};
use egui::{ClippedPrimitive, Context, FullOutput, PlatformOutput, TextureId, TexturesDelta};
use egui_winit::{EventResponse, State};
use glam::Mat4;
use inline_spirv::include_spirv;
use std::collections::{hash_map::Entry, HashMap};
use std::mem::size_of;
use std::slice;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::event::WindowEvent;
use winit::event_loop::EventLoopWindowTarget;
use winit::window::Window;

pub use egui; //FIXME: forward only what's needed to build UIs

const MIN_FRAME_TIME: Duration = Duration::from_micros(1_000_000 / 60);

pub struct UiRenderer {
    device: Arc<VulkanDevice>,
    context: Context,
    winit_state: State,
    frame_output: Option<FullOutput>,
    platform_output: Option<PlatformOutput>,
    textures: HashMap<TextureId, Texture>,
    pipeline: Pipeline,
    set_layout: vk::DescriptorSetLayout,
    buffer: VkBuffer,
    cmd_buffer: Option<vk::CommandBuffer>,
    last_draw_time: Option<Instant>,
}

impl UiRenderer {
    pub fn new(event_loop: &EventLoopWindowTarget<()>, engine: &VulkanEngine) -> VulkanResult<Self> {
        let device = engine.device.clone();
        let mut shader = Shader::new(
            &device,
            include_spirv!("src/shaders/gui.vert.glsl", vert, glsl),
            include_spirv!("src/shaders/gui.frag.glsl", frag, glsl),
        )?;
        let sampler = engine.get_sampler(vk::Filter::LINEAR, vk::Filter::LINEAR, vk::SamplerAddressMode::CLAMP_TO_EDGE)?;
        let set_layout = device.create_descriptor_set_layout(&[
            // layout(binding = 0) uniform sampler2D texSampler
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .immutable_samplers(slice::from_ref(&sampler))
                .build(),
        ])?;
        let push_constants = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<Mat4>() as _);
        let pipeline = Pipeline::builder(&shader)
            .vertex_input::<Vertex>()
            .descriptor_layout(set_layout)
            .push_constants(slice::from_ref(&push_constants))
            .render_to_swapchain(&engine.swapchain)
            .mode(PipelineMode::Overlay)
            .build(&device)?;
        unsafe { shader.cleanup(&device) };
        let buffer = device.allocate_buffer(
            65536,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            ga::UsageFlags::UPLOAD,
        )?;

        Ok(Self {
            device,
            context: egui::Context::default(),
            winit_state: State::new(event_loop),
            frame_output: None,
            platform_output: None,
            textures: Default::default(),
            pipeline,
            set_layout,
            buffer,
            cmd_buffer: None,
            last_draw_time: None,
        })
    }

    pub fn event_input(&mut self, event: &WindowEvent) -> EventResponse {
        self.winit_state.on_event(&self.context, event)
    }

    pub fn run(&mut self, window: &Window, run_ui: impl FnOnce(&Context)) {
        let now = Instant::now();
        if let Some(last_draw) = self.last_draw_time {
            if now - last_draw < MIN_FRAME_TIME {
                return;
            }
        }
        self.last_draw_time = Some(now);
        let raw_input = self.winit_state.take_egui_input(window);
        self.frame_output = self.context.run(raw_input, run_ui).into();
    }

    pub fn draw(&mut self, engine: &VulkanEngine) -> VulkanResult<DrawPayload> {
        let Some(run_output) = self.frame_output.take() else {
            return Ok(DrawPayload::new(self.cmd_buffer.expect("draw called before run"), false));
        };

        self.platform_output = Some(run_output.platform_output);
        let primitives = self.context.tessellate(run_output.shapes);
        let mut drop_textures = self.update_textures(run_output.textures_delta)?;

        let cmd_buffer = engine.create_secondary_command_buffer()?;
        let old_cmdbuf = self.cmd_buffer.replace(cmd_buffer);
        let mut drop_buffers = self.build_draw_commands(cmd_buffer, primitives, engine)?;

        let pool = engine.secondary_cmd_pool; //FIXME: use a separate command pool
        let payload = DrawPayload::new_with_callback(cmd_buffer, false, move |dev| unsafe {
            drop_buffers.cleanup(dev);
            drop_textures.cleanup(dev);
            if let Some(cmdbuf) = old_cmdbuf {
                dev.free_command_buffers(pool, &[cmdbuf]);
            }
        });
        Ok(payload)
    }

    pub fn event_output(&mut self, window: &Window) {
        if let Some(po) = self.platform_output.take() {
            self.winit_state.handle_platform_output(window, &self.context, po);
        }
    }

    fn update_textures(&mut self, tex_delta: TexturesDelta) -> VulkanResult<Vec<Texture>> {
        // create or update textures
        for (id, image) in tex_delta.set {
            let (pixels, [width, height]) = match image.image {
                egui::ImageData::Color(img) => (img.pixels, img.size.map(|v| v as _)),
                egui::ImageData::Font(img) => (img.srgba_pixels(None).collect(), img.size.map(|v| v as _)),
            };
            let bytes = bytemuck::cast_slice(&pixels);
            match self.textures.entry(id) {
                Entry::Vacant(entry) => {
                    entry.insert(Texture::new(
                        &self.device,
                        width,
                        height,
                        vk::Format::R8G8B8A8_SRGB,
                        bytes,
                        vk::Sampler::null(),
                    )?);
                }
                Entry::Occupied(mut entry) => {
                    let [x, y] = image.pos.unwrap_or([0, 0]).map(|v| v as _);
                    entry.get_mut().update(&self.device, x, y, width, height, bytes)?;
                }
            }
        }
        // return textures to be deleted in next frame
        let drop_textures: Vec<_> = tex_delta.free.iter().filter_map(|tex_id| self.textures.remove(tex_id)).collect();
        Ok(drop_textures)
    }

    fn build_draw_commands(
        &mut self, cmd_buffer: vk::CommandBuffer, primitives: Vec<ClippedPrimitive>, engine: &VulkanEngine,
    ) -> VulkanResult<Vec<VkBuffer>> {
        // get combined buffer size for a single shared allocation
        let (total_verts, total_idx) = primitives.iter().fold((0, 0), |acc, prim| match prim.primitive {
            Primitive::Mesh(ref mesh) => (acc.0 + mesh.vertices.len(), acc.1 + mesh.indices.len()),
            _ => acc,
        });
        let total_vert_size = total_verts * size_of::<Vertex>();
        let total_bytes = total_vert_size + total_idx * size_of::<u32>();
        // allocate nearest power of two sized buffer if necessary
        let device = &*self.device;
        let mut drop_buffers = vec![];
        if total_bytes as u64 > self.buffer.memory.size() {
            let new_size = 1u64 << ((total_bytes - 1).ilog2() + 1);
            let new_buffer = device.allocate_buffer(
                new_size,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
                ga::UsageFlags::UPLOAD,
            )?;
            drop_buffers.push(std::mem::replace(&mut self.buffer, new_buffer));
        }
        // build a command buffer from the primitives
        engine.begin_secondary_draw_commands(cmd_buffer, vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?;
        unsafe {
            device.debug(|d| d.cmd_begin_label(cmd_buffer, "UI", [0.6, 0.2, 0.4, 1.0]));
            device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, *self.pipeline);
            device.cmd_bind_vertex_buffers(cmd_buffer, 0, &[*self.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd_buffer, *self.buffer, total_vert_size as _, vk::IndexType::UINT32);
            device.cmd_set_viewport(cmd_buffer, 0, &[engine.swapchain.viewport()]);
        }
        let mut vert_offset = 0;
        let mut idx_offset = 0;
        let idx_base = total_vert_size / 4;
        let mut mem = self.buffer.map(device)?;
        for prim in primitives {
            match prim.primitive {
                Primitive::Mesh(mesh) => {
                    let n_vert = mesh.vertices.len();
                    let n_idx = mesh.indices.len();
                    mem.write_slice(&mesh.vertices, vert_offset);
                    mem.write_slice(&mesh.indices, idx_base + idx_offset);
                    let vk::Extent2D { width, height } = engine.swapchain.extent;
                    let proj = Mat4::orthographic_rh(0.0, width as _, 0.0, height as _, 0.0, 1.0);
                    let texture = self.textures.get(&mesh.texture_id).describe_err("Missing gui texture")?;
                    let image_info = texture.descriptor();
                    let desc_writes = vk::WriteDescriptorSet::builder()
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(slice::from_ref(&image_info));
                    unsafe {
                        device.cmd_set_scissor(cmd_buffer, 0, &[vk_rect(prim.clip_rect)]);
                        device.pushdesc_fn.cmd_push_descriptor_set(
                            cmd_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline.layout,
                            0,
                            slice::from_ref(&desc_writes),
                        );
                        device.cmd_push_constants(
                            cmd_buffer,
                            self.pipeline.layout,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            bytemuck::bytes_of(&proj),
                        );
                        device.cmd_draw_indexed(cmd_buffer, n_idx as _, 1, idx_offset as _, vert_offset as _, 0);
                    }
                    vert_offset += n_vert;
                    idx_offset += n_idx;
                }
                Primitive::Callback(cb) => {
                    eprintln!("Unimplemented: paint callback: {cb:?}");
                }
            }
        }
        device.debug(|d| d.cmd_end_label(cmd_buffer));
        engine.end_secondary_draw_commands(cmd_buffer)?;
        Ok(drop_buffers)
    }
}

impl Drop for UiRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.textures.cleanup(&self.device);
            self.pipeline.cleanup(&self.device);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
            self.buffer.cleanup(&self.device);
        }
    }
}

fn vk_rect(rect: egui::Rect) -> vk::Rect2D {
    let size = (rect.max - rect.min).abs();
    vk::Rect2D {
        offset: vk::Offset2D {
            x: rect.min.x as _,
            y: rect.min.y as _,
        },
        extent: vk::Extent2D {
            width: size.x as _,
            height: size.y as _,
        },
    }
}
