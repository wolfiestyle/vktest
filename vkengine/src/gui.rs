use crate::create::{CreateFromInfo, VkOwned};
use crate::device::{ImageData, VulkanDevice};
use crate::engine::{CmdBufferRing, DrawPayload, UploadBuffer, VulkanEngine};
use crate::pipeline::{Pipeline, PipelineMode, Shader};
use crate::texture::{Texture, TextureOptions};
use crate::types::VulkanResult;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use egui::epaint::{ImageDelta, Primitive, Vertex};
use egui::{ClippedPrimitive, Context, FullOutput, PlatformOutput, TextureId, TexturesDelta, Ui, ViewportId};
use egui_winit::{EventResponse, State};
use glam::{Mat4, camera::rh::proj::vulkan};
use inline_spirv::include_spirv;
use std::collections::{HashMap, HashSet};
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
    textures_to_free: HashSet<TextureId>,
    shader: Shader,
    pipeline: Pipeline,
    set_layout: VkOwned<vk::DescriptorSetLayout>,
    push_constants: vk::PushConstantRange,
    buffers: UploadBuffer,
    buffers_dirty: bool,
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
            .create_owned(&device)?;
        device.debug(|d| d.set_object_name(*set_layout, "UiRenderer desc layout"));
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
        let winit_state = State::new(context.clone(), ViewportId::ROOT, &window, None, None, None);

        Ok(Self {
            device,
            context,
            winit_state,
            frame_output: None,
            platform_output: None,
            primitives: vec![],
            textures: Default::default(),
            textures_to_free: HashSet::new(),
            shader,
            pipeline,
            set_layout,
            push_constants,
            buffers,
            buffers_dirty: false,
            local_frame: 0,
            total_vert_size: 0,
            cmd_buffers,
            last_draw_time: None,
        })
    }

    pub fn event_input(&mut self, window: &Window, event: &WindowEvent) -> EventResponse {
        self.winit_state.on_window_event(window, event)
    }

    pub fn run(&mut self, window: &Window, run_ui: impl FnMut(&mut Ui)) {
        let now = Instant::now();
        if self.last_draw_time.is_some_and(|last_draw| now - last_draw < MIN_FRAME_TIME) {
            return;
        }
        self.last_draw_time = Some(now);
        let raw_input = self.winit_state.take_egui_input(window);
        let output = self.context.run_ui(raw_input, run_ui);
        if let Some(pending) = &mut self.frame_output {
            pending.append(output);
        } else {
            self.frame_output = Some(output);
        }
    }

    pub fn draw(&mut self, engine: &VulkanEngine) -> VulkanResult<DrawPayload> {
        if let Some(output) = &mut self.frame_output {
            let device = &self.device;
            let textures = &mut self.textures;
            apply_texture_updates(&mut output.textures_delta, |id, image| {
                Self::update_texture(device, textures, id, image)
            })?;
        }
        if let Some(mut output) = self.frame_output.take() {
            self.local_frame += 1;
            self.platform_output = Some(output.platform_output);
            self.primitives = self.context.tessellate(output.shapes, output.pixels_per_point);
            self.textures_to_free.extend(output.textures_delta.free.drain());
            self.buffers_dirty = true;
        }
        if self.buffers_dirty {
            self.update_buffers()?;
            self.buffers_dirty = false;
        }

        let cmd_buffer = self.cmd_buffers.get_current_buffer(engine)?;
        self.build_draw_commands(cmd_buffer, engine)?;
        if self.textures_to_free.is_empty() {
            return Ok(DrawPayload::new(cmd_buffer));
        }

        // Draws between egui frames reuse these primitives, so keep their textures available.
        let referenced_textures: HashSet<_> = self
            .primitives
            .iter()
            .filter_map(|primitive| match &primitive.primitive {
                Primitive::Mesh(mesh) => Some(mesh.texture_id),
                Primitive::Callback(_) => None,
            })
            .collect();
        let mut drop_textures = Vec::new();
        self.textures_to_free.retain(|id| {
            let referenced = referenced_textures.contains(id);
            if !referenced {
                drop_textures.extend(self.textures.remove(id));
            }
            referenced
        });
        if drop_textures.is_empty() {
            Ok(DrawPayload::new(cmd_buffer))
        } else {
            Ok(DrawPayload::new_with_callback(cmd_buffer, move |_dev| drop(drop_textures)))
        }
    }

    pub fn event_output(&mut self, window: &Window) {
        if let Some(po) = self.platform_output.take() {
            self.winit_state.handle_platform_output(window, po);
        }
    }

    fn update_texture(
        device: &Arc<VulkanDevice>, textures: &mut HashMap<TextureId, Texture>, id: TextureId, image: &ImageDelta,
    ) -> VulkanResult<()> {
        let egui::ImageData::Color(img) = &image.image;
        let bytes = bytemuck::cast_slice::<egui::Color32, u8>(&img.pixels);
        let size = img.size.map(|v| v as u32).into();
        match image.pos {
            None => {
                let texture = Texture::new(
                    device,
                    size,
                    vk::Format::R8G8B8A8_SRGB,
                    ImageData::Single(bytes),
                    vk::Sampler::null(),
                    TextureOptions {
                        gen_mipmaps: false,
                        swizzle: None,
                    },
                )?;
                textures.insert(id, texture);
            }
            Some(pos) => {
                if let Some(tex) = textures.get_mut(&id) {
                    tex.update(pos.map(|n| n as u32).into(), size, bytes)?;
                } else {
                    eprintln!("Error: received ImageDelta update on undefined texture {id:?}");
                }
            }
        }
        Ok(())
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
        let resized = buffer.ensure_capacity(total_bytes as _, false)?;
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
        let scale = self.context.pixels_per_point();
        let constants = PushConstants {
            proj: gui_projection(engine.swapchain.extent),
            scale,
            _unused: Default::default(),
        };
        unsafe {
            device.cmd_push_constants(
                cmd_buffer,
                *self.pipeline.layout,
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
                            *self.pipeline.layout,
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
        self.pipeline = pipeline;
        Ok(())
    }
}

impl Drop for UiRenderer {
    fn drop(&mut self) {
        if let Some(output) = self.frame_output.take() {
            output.drop_without_applying_deltas();
        }
        unsafe {
            self.device.device_wait_idle().unwrap();
        }
    }
}

fn apply_texture_updates(delta: &mut TexturesDelta, mut apply: impl FnMut(TextureId, &ImageDelta) -> VulkanResult<()>) -> VulkanResult<()> {
    for (&id, images) in &mut delta.set {
        let mut applied = 0;
        let result: VulkanResult<()> = images.iter().try_for_each(|image| {
            apply(id, image)?;
            applied += 1;
            Ok(())
        });
        // Preserve the failed update and any later patches for the next draw.
        images.drain(..applied);
        result?;
    }
    delta.set.clear();
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstants {
    proj: Mat4,
    scale: f32,
    _unused: [f32; 3],
}

fn gui_projection(extent: vk::Extent2D) -> Mat4 {
    // GUI pixels start at the top, so pass the bottom bound before the top bound.
    vulkan::orthographic(0.0, extent.width as f32, extent.height as f32, 0.0, 0.0, 1.0)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VkError;
    use glam::Vec3;

    #[test]
    fn gui_projection_maps_pixels_top_to_bottom_in_the_positive_height_viewport() {
        let projection = gui_projection(vk::Extent2D { width: 800, height: 600 });

        for (pixel, ndc) in [
            (Vec3::ZERO, Vec3::new(-1.0, -1.0, 0.0)),
            (Vec3::new(400.0, 300.0, 0.0), Vec3::ZERO),
            (Vec3::new(800.0, 600.0, 0.0), Vec3::new(1.0, 1.0, 0.0)),
        ] {
            let projected = projection.project_point3(pixel);
            assert!(
                projected.abs_diff_eq(ndc, 1e-5),
                "expected {pixel:?} to project to {ndc:?}, got {projected:?}"
            );
        }
    }

    #[test]
    fn applies_texture_patches_in_order_when_ui_frames_are_combined() {
        let id = TextureId::Managed(1);
        let mut pending = FullOutput::default();
        pending.textures_delta.push(
            id,
            ImageDelta::full(egui::ColorImage::filled([2, 2], egui::Color32::BLACK), egui::TextureOptions::LINEAR),
        );
        let mut next = FullOutput::default();
        for pos in [[1, 0], [1, 1]] {
            next.textures_delta.push(
                id,
                ImageDelta::partial(
                    pos,
                    egui::ColorImage::filled([1, 1], egui::Color32::WHITE),
                    egui::TextureOptions::LINEAR,
                ),
            );
        }
        pending.append(next);

        let mut applied = Vec::new();
        apply_texture_updates(&mut pending.textures_delta, |id, delta| {
            applied.push((id, delta.pos));
            Ok(())
        })
        .unwrap();

        assert_eq!(applied, [(id, None), (id, Some([1, 0])), (id, Some([1, 1]))]);
        assert!(pending.textures_delta.is_empty());
    }

    #[test]
    fn retries_texture_uploads_from_the_failed_patch() {
        let id = TextureId::Managed(1);
        let mut delta = TexturesDelta::default();
        for pos in [[0, 0], [1, 0], [1, 1]] {
            delta.push(
                id,
                ImageDelta::partial(
                    pos,
                    egui::ColorImage::filled([1, 1], egui::Color32::WHITE),
                    egui::TextureOptions::LINEAR,
                ),
            );
        }
        let first_attempt = apply_texture_updates(&mut delta, |_, image| {
            if image.pos == Some([1, 0]) {
                return Err(VkError::InvalidArgument("test upload failure"));
            }
            Ok(())
        });
        let mut applied_on_retry = Vec::new();
        apply_texture_updates(&mut delta, |_, image| {
            applied_on_retry.push(image.pos);
            Ok(())
        })
        .unwrap();

        assert!(first_attempt.is_err());
        assert_eq!(applied_on_retry, [Some([1, 0]), Some([1, 1])]);
    }

    #[test]
    fn bundled_fonts_produce_an_atlas_upload_and_text_mesh() {
        let context = Context::default();
        let mut output = context.run_ui(egui::RawInput::default(), |ui| {
            ui.label("Vulkan test");
        });
        let mut uploaded_atlas = false;
        apply_texture_updates(&mut output.textures_delta, |id, image| {
            uploaded_atlas |= id == TextureId::default() && image.pos.is_none();
            Ok(())
        })
        .unwrap();
        let primitives = context.tessellate(output.shapes, output.pixels_per_point);
        output.textures_delta.clear();

        assert!(uploaded_atlas);
        assert!(primitives.iter().any(|primitive| {
            matches!(&primitive.primitive, Primitive::Mesh(mesh) if mesh.texture_id == TextureId::default() && !mesh.indices.is_empty())
        }));
    }
}
