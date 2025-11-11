use crate::create::CreateFromInfo;
use crate::device::{ImageData, VulkanDevice};
use crate::engine::{CmdBufferRing, DrawPayload, UploadBuffer, VulkanEngine};
use crate::pipeline::{Pipeline, PipelineMode, Shader};
use crate::texture::{Texture, TextureOptions};
use crate::types::{Cleanup, VulkanResult};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use egui::epaint::{Primitive, Vertex};
use egui::{ClippedPrimitive, Context, FullOutput, PlatformOutput, TextureId, TexturesDelta, ViewportId};
use egui_winit::{EventResponse, State};
use glam::Mat4;
use inline_spirv::include_spirv;
use std::collections::{HashMap, hash_map::Entry};
use std::mem::size_of;
use std::slice;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::event::WindowEvent;
use winit::window::Window;

pub use egui; //FIXME: forward only what's needed to build UIs

const MIN_FRAME_TIME: Duration = Duration::from_micros(1_000_000 / 60);
const INITIAL_BUFFER_SIZE: u64 = 65536;

pub struct UiRenderer {
    device: Arc<VulkanDevice>,
    context: Context,
    winit_state: State,
    frame_output: Option<FullOutput>,
    platform_output: Option<PlatformOutput>,
    primitives: Vec<ClippedPrimitive>,
    textures: HashMap<TextureId, Texture>,
    shader: Shader,
    pipeline: Pipeline,
    set_layout: vk::DescriptorSetLayout,
    push_constants: vk::PushConstantRange,
    buffers: UploadBuffer,
    local_frame: u64,
    total_vert_size: vk::DeviceSize,
    cmd_buffers: CmdBufferRing,
    last_draw_time: Option<Instant>,
}

impl UiRenderer {
    pub fn new(window: &Window, engine: &VulkanEngine) -> VulkanResult<Self> {
        let device = engine.device.clone();
        let shader = Shader::new(
            &device,
            include_spirv!("src/shaders/gui.vert.glsl", vert, glsl),
            include_spirv!("src/shaders/gui.frag.glsl", frag, glsl),
        )?;
        let sampler = engine.get_sampler(vk::SamplerAddressMode::CLAMP_TO_EDGE.into())?;
        let set_layout = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&[vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .immutable_samplers(slice::from_ref(&sampler))])
            .create(&device)?;
        let push_constants = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<PushConstants>() as _);
        let pipeline = Pipeline::builder_graphics(&shader)
            .vertex_input::<Vertex>()
            .descriptor_layout(&set_layout)
            .push_constants(slice::from_ref(&push_constants))
            .render_to_swapchain(&engine.swapchain)
            .mode(PipelineMode::Overlay)
            .build(engine)?;

        let buffers = UploadBuffer::new(
            &device,
            INITIAL_BUFFER_SIZE,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            "UiRenderer buffer",
        )?;

        let cmd_buffers = CmdBufferRing::new(&device)?;

        let context = egui::Context::default();
        let winit_state = State::new(context.clone(), ViewportId::ROOT, &window, None, None);

        Ok(Self {
            device,
            context,
            winit_state,
            frame_output: None,
            platform_output: None,
            primitives: vec![],
            textures: Default::default(),
            shader,
            pipeline,
            set_layout,
            push_constants,
            buffers,
            local_frame: 0,
            total_vert_size: 0,
            cmd_buffers,
            last_draw_time: None,
        })
    }

    pub fn event_input(&mut self, window: &Window, event: &WindowEvent) -> EventResponse {
        self.winit_state.on_window_event(window, event)
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
        if let Some(run_output) = self.frame_output.take() {
            // process a new set of primitives
            self.local_frame += 1;
            self.platform_output = Some(run_output.platform_output);
            self.primitives = self.context.tessellate(run_output.shapes, run_output.pixels_per_point);
            let drop_textures = self.update_textures(run_output.textures_delta)?;
            self.update_buffers()?;

            let cmd_buffer = self.cmd_buffers.get_current_buffer(engine)?;
            self.build_draw_commands(cmd_buffer, engine)?;
            let payload = DrawPayload::new_with_callback(cmd_buffer, |dev| dev.dispose_of(drop_textures));
            Ok(payload)
        } else {
            // build a command buffer from the old primitives
            let cmd_buffer = self.cmd_buffers.get_current_buffer(engine)?;
            self.build_draw_commands(cmd_buffer, engine)?;
            Ok(DrawPayload::new(cmd_buffer))
        }
    }

    pub fn event_output(&mut self, window: &Window) {
        if let Some(po) = self.platform_output.take() {
            self.winit_state.handle_platform_output(window, po);
        }
    }

    fn update_textures(&mut self, tex_delta: TexturesDelta) -> VulkanResult<Vec<Texture>> {
        // create or update textures
        for (id, image) in tex_delta.set {
            let (bytes, size, format) = match &image.image {
                egui::ImageData::Color(img) => (
                    bytemuck::cast_slice::<egui::Color32, u8>(&img.pixels),
                    img.size.map(|v| v as u32),
                    vk::Format::R8G8B8A8_SRGB,
                ),
                egui::ImageData::Font(img) => (
                    bytemuck::cast_slice::<f32, u8>(&img.pixels),
                    img.size.map(|v| v as u32),
                    vk::Format::R32_SFLOAT,
                ),
            };
            match self.textures.entry(id) {
                Entry::Vacant(entry) => {
                    let swizzle = (format == vk::Format::R32_SFLOAT).then_some(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::R,
                        b: vk::ComponentSwizzle::R,
                        a: vk::ComponentSwizzle::R,
                    });
                    entry.insert(Texture::new(
                        &self.device,
                        size.into(),
                        format,
                        ImageData::Single(bytes),
                        vk::Sampler::null(),
                        TextureOptions {
                            gen_mipmaps: false,
                            swizzle,
                        },
                    )?);
                }
                Entry::Occupied(mut entry) => {
                    let pos = image.pos.unwrap_or([0, 0]).map(|v| v as _);
                    entry.get_mut().update(&self.device, pos.into(), size.into(), bytes)?;
                }
            }
        }
        // return textures to be deleted in next frame
        let drop_textures = tex_delta
            .free
            .into_iter()
            .filter_map(|tex_id| self.textures.remove(&tex_id))
            .collect();
        Ok(drop_textures)
    }

    fn update_buffers(&mut self) -> VulkanResult<()> {
        // get combined buffer size for a single shared allocation
        let (total_verts, total_idx) = self.primitives.iter().fold((0, 0), |acc, prim| match &prim.primitive {
            Primitive::Mesh(mesh) => (acc.0 + mesh.vertices.len(), acc.1 + mesh.indices.len()),
            _ => acc,
        });
        let total_vert_size = total_verts * size_of::<Vertex>();
        let total_bytes = total_vert_size + total_idx * size_of::<u32>();
        self.total_vert_size = total_vert_size as _;
        // allocate more buffer space if necessary
        let buffer = &mut self.buffers[self.local_frame];
        let resized = buffer.ensure_capacity(&self.device, total_bytes as _, false)?;
        if resized {
            self.device.debug(|d| d.set_object_name(buffer.handle, "UiRenderer buffer"))
        }
        // write vertices and indices on the same buffer
        let mut vert_offset = 0;
        let mut idx_offset = total_vert_size / size_of::<u32>();
        let mut mem = buffer.map()?;
        for prim in &self.primitives {
            match &prim.primitive {
                Primitive::Mesh(mesh) => {
                    mem.write_slice(&mesh.vertices, vert_offset);
                    mem.write_slice(&mesh.indices, idx_offset);
                    vert_offset += mesh.vertices.len();
                    idx_offset += mesh.indices.len();
                }
                Primitive::Callback(_) => (),
            }
        }
        Ok(())
    }

    fn build_draw_commands(&mut self, cmd_buffer: vk::CommandBuffer, engine: &VulkanEngine) -> VulkanResult<()> {
        let device = &*self.device;
        engine.begin_secondary_draw_commands(cmd_buffer, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
        let buffer = self.buffers[self.local_frame].handle;
        unsafe {
            device.debug(|d| d.cmd_begin_label(cmd_buffer, "UI", [0.6, 0.2, 0.4, 1.0]));
            device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, *self.pipeline);
            device.cmd_bind_vertex_buffers(cmd_buffer, 0, &[buffer], &[0]);
            device.cmd_bind_index_buffer(cmd_buffer, buffer, self.total_vert_size, vk::IndexType::UINT32);
            device.cmd_set_viewport(cmd_buffer, 0, &[engine.swapchain.viewport()]);
        }
        let vk::Extent2D { width, height } = engine.swapchain.extent;
        let scale = self.context.pixels_per_point();
        let constants = PushConstants {
            proj: Mat4::orthographic_rh(0.0, width as _, 0.0, height as _, 0.0, 1.0),
            scale,
            _unused: Default::default(),
        };
        unsafe {
            device.cmd_push_constants(
                cmd_buffer,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&constants),
            );
        }

        let mut vert_offset = 0;
        let mut idx_offset = 0;
        for prim in &self.primitives {
            unsafe { device.cmd_set_scissor(cmd_buffer, 0, &[vk_rect(prim.clip_rect, scale)]) };
            match &prim.primitive {
                Primitive::Mesh(mesh) => {
                    let n_vert = mesh.vertices.len() as i32;
                    let n_idx = mesh.indices.len() as u32;
                    let texture = self.textures.get(&mesh.texture_id).unwrap_or(&engine.default_texture);
                    let desc_writes = vk::WriteDescriptorSet::default()
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(slice::from_ref(&texture.info));
                    unsafe {
                        device.pushdesc_fn.cmd_push_descriptor_set(
                            cmd_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline.layout,
                            0,
                            slice::from_ref(&desc_writes),
                        );
                        device.cmd_draw_indexed(cmd_buffer, n_idx, 1, idx_offset, vert_offset, 0);
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
        Ok(())
    }

    pub fn rebuild_pipeline(&mut self, engine: &VulkanEngine) -> VulkanResult<()> {
        let pipeline = Pipeline::builder_graphics(&self.shader)
            .vertex_input::<Vertex>()
            .descriptor_layout(&self.set_layout)
            .push_constants(slice::from_ref(&self.push_constants))
            .render_to_swapchain(&engine.swapchain)
            .mode(PipelineMode::Overlay)
            .build(engine)?;
        let old_pipeline = std::mem::replace(&mut self.pipeline, pipeline);
        self.device.dispose_of(old_pipeline);
        Ok(())
    }
}

impl Drop for UiRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.textures.cleanup(&self.device);
            self.pipeline.cleanup(&self.device);
            self.shader.cleanup(&self.device);
            self.set_layout.cleanup(&self.device);
            self.buffers.cleanup(&self.device);
            self.cmd_buffers.cleanup(&self.device);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstants {
    proj: Mat4,
    scale: f32,
    _unused: [f32; 3],
}

fn vk_rect(rect: egui::Rect, scale: f32) -> vk::Rect2D {
    let pos = rect.min * scale;
    let size = (rect.max - rect.min).abs() * scale;
    vk::Rect2D {
        offset: vk::Offset2D {
            x: pos.x as _,
            y: pos.y as _,
        },
        extent: vk::Extent2D {
            width: size.x as _,
            height: size.y as _,
        },
    }
}
