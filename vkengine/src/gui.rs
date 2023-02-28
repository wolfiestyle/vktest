use crate::device::{VkBuffer, VulkanDevice};
use crate::engine::{Pipeline, PipelineMode, Shader, Texture, VulkanEngine};
use crate::types::{Cleanup, ErrorDescription, VulkanResult};
use ash::vk;
use egui::epaint::{Primitive, Vertex};
use egui::{ClippedPrimitive, Context, FullOutput, PlatformOutput, TextureFilter, TextureId, TextureOptions, TexturesDelta};
use egui_winit::{EventResponse, State};
use glam::Mat4;
use inline_spirv::include_spirv;
use std::collections::{hash_map::Entry, HashMap};
use std::mem::size_of;
use std::slice;
use winit::event::WindowEvent;
use winit::event_loop::EventLoopWindowTarget;
use winit::window::Window;

pub use egui; //FIXME: forward only what's needed to build UIs

pub struct VkGui {
    context: Context,
    winit_state: State,
    platform_output: Option<PlatformOutput>,
    textures: HashMap<TextureId, TextureSlot>,
    samplers: HashMap<TextureOptions, vk::Sampler>,
    pipeline: Pipeline,
    pipeline_layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    buffer: VkBuffer,
    deletion_pending: bool,
}

impl VkGui {
    pub fn new(event_loop: &EventLoopWindowTarget<()>, engine: &VulkanEngine) -> VulkanResult<Self> {
        let device = &engine.device;
        let mut shader = Shader::new(
            device,
            include_spirv!("src/shaders/gui.vert.glsl", vert, glsl),
            include_spirv!("src/shaders/gui.frag.glsl", frag, glsl),
        )?;
        let set_layout = device.create_descriptor_set_layout(&[
            // layout(binding = 0) uniform sampler2D texSampler
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
        ])?;
        let push_constants = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<Mat4>() as _);
        let pipeline_layout = device.create_pipeline_layout(slice::from_ref(&set_layout), slice::from_ref(&push_constants))?;
        let pipeline = Pipeline::new::<Vertex>(device, &shader, pipeline_layout, &engine.swapchain, PipelineMode::Overlay)?;
        unsafe { shader.cleanup(device) };
        let buffer = device.allocate_buffer(
            65536,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            ga::UsageFlags::UPLOAD,
        )?;
        Ok(Self {
            context: egui::Context::default(),
            winit_state: State::new(event_loop),
            platform_output: None,
            textures: Default::default(),
            samplers: Default::default(),
            pipeline,
            pipeline_layout,
            set_layout,
            buffer,
            deletion_pending: false,
        })
    }

    pub fn event_input(&mut self, event: &WindowEvent) -> EventResponse {
        self.winit_state.on_event(&self.context, event)
    }

    pub fn run(&mut self, window: &Window, run_ui: impl FnOnce(&Context)) -> FullOutput {
        let raw_input = self.winit_state.take_egui_input(window);
        self.context.run(raw_input, run_ui)
    }

    pub fn draw(&mut self, run_output: FullOutput, engine: &VulkanEngine) -> VulkanResult<vk::CommandBuffer> {
        let primitives = self.context.tessellate(run_output.shapes);
        self.cleanup_textures(&engine.device)?;
        self.update_textures(run_output.textures_delta, &engine.device)?;
        self.platform_output = Some(run_output.platform_output);
        self.build_draw_commands(primitives, engine)
    }

    pub fn event_output(&mut self, window: &Window) {
        if let Some(po) = self.platform_output.take() {
            self.winit_state.handle_platform_output(window, &self.context, po);
        }
    }

    fn update_textures(&mut self, tex_delta: TexturesDelta, device: &VulkanDevice) -> VulkanResult<()> {
        // create or update textures
        for (id, image) in tex_delta.set {
            let (pixels, [width, height]) = match image.image {
                egui::ImageData::Color(img) => (img.pixels, img.size.map(|v| v as _)),
                egui::ImageData::Font(img) => (img.srgba_pixels(None).collect(), img.size.map(|v| v as _)),
            };
            let bytes = bytemuck::cast_slice(&pixels);
            match self.textures.entry(id) {
                Entry::Vacant(entry) => {
                    let sampler = match self.samplers.entry(image.options) {
                        Entry::Occupied(entry) => *entry.get(),
                        Entry::Vacant(entry) => {
                            let sampler = device.create_texture_sampler(
                                vk_filter(image.options.magnification),
                                vk_filter(image.options.minification),
                                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            )?;
                            entry.insert(sampler);
                            sampler
                        }
                    };
                    entry.insert(TextureSlot {
                        texture: Texture::new(device, width, height, vk::Format::R8G8B8A8_SRGB, bytes, sampler)?,
                        delete: false,
                    });
                }
                Entry::Occupied(mut entry) => {
                    let [x, y] = image.pos.unwrap_or([0, 0]).map(|v| v as _);
                    entry.get_mut().texture.update(device, x, y, width, height, bytes)?;
                }
            }
        }
        // mark textures to be deleted in next frame
        if !tex_delta.free.is_empty() {
            self.deletion_pending = true;
        }
        for id in tex_delta.free {
            if let Some(ts) = self.textures.get_mut(&id) {
                ts.delete = true;
            }
        }
        Ok(())
    }

    fn cleanup_textures(&mut self, device: &VulkanDevice) -> VulkanResult<()> {
        if self.deletion_pending {
            unsafe {
                device.queue_wait_idle(device.graphics_queue)?; //FIXME: sync properly
            }
            self.textures.retain(|_, ts| {
                if ts.delete {
                    unsafe { ts.texture.cleanup(device) };
                }
                !ts.delete
            });
            self.deletion_pending = false;
        }
        Ok(())
    }

    fn build_draw_commands(&mut self, primitives: Vec<ClippedPrimitive>, engine: &VulkanEngine) -> VulkanResult<vk::CommandBuffer> {
        // get combined buffer size for a single shared allocation
        let (total_verts, total_idx) = primitives.iter().fold((0, 0), |acc, prim| match prim.primitive {
            Primitive::Mesh(ref mesh) => (acc.0 + mesh.vertices.len(), acc.1 + mesh.indices.len()),
            _ => acc,
        });
        let total_vert_size = total_verts * size_of::<Vertex>();
        let total_bytes = total_vert_size + total_idx * size_of::<u32>();
        // allocate nearest power of two sized buffer if necessary
        let device = &engine.device;
        if total_bytes as u64 > self.buffer.memory.size() {
            let new_size = 1u64 << ((total_bytes - 1).ilog2() + 1);
            unsafe {
                device.queue_wait_idle(device.graphics_queue)?; //FIXME: sync properly
                self.buffer.cleanup(device);
            }
            self.buffer = device.allocate_buffer(
                new_size,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
                ga::UsageFlags::UPLOAD,
            )?;
        }
        // build a command buffer from the primitives
        let cmd_buffer = engine.begin_secondary_draw_commands()?;
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
        for prim in primitives {
            match prim.primitive {
                Primitive::Mesh(mesh) => {
                    let n_vert = mesh.vertices.len();
                    let n_idx = mesh.indices.len();
                    self.buffer.write_slice(device, &mesh.vertices, vert_offset)?;
                    self.buffer.write_slice(device, &mesh.indices, idx_base + idx_offset)?;
                    let vk::Extent2D { width, height } = engine.swapchain.extent;
                    let proj = Mat4::orthographic_rh(0.0, width as _, 0.0, height as _, 0.0, 1.0);
                    let texture = &self.textures.get(&mesh.texture_id).describe_err("Missing gui texture")?.texture;
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
                            self.pipeline_layout,
                            0,
                            slice::from_ref(&desc_writes),
                        );
                        device.cmd_push_constants(
                            cmd_buffer,
                            self.pipeline_layout,
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
        engine.end_secondary_draw_commands(cmd_buffer)
    }
}

impl Cleanup<VulkanDevice> for VkGui {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        device.device_wait_idle().unwrap();
        self.textures.cleanup(device);
        for &sampler in self.samplers.values() {
            device.destroy_sampler(sampler, None);
        }
        self.pipeline.cleanup(device);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_descriptor_set_layout(self.set_layout, None);
        self.buffer.cleanup(device);
    }
}

struct TextureSlot {
    texture: Texture,
    delete: bool,
}

impl Cleanup<VulkanDevice> for TextureSlot {
    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        self.texture.cleanup(device)
    }
}

fn vk_filter(filter: TextureFilter) -> vk::Filter {
    match filter {
        TextureFilter::Nearest => vk::Filter::NEAREST,
        TextureFilter::Linear => vk::Filter::LINEAR,
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
