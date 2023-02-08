use crate::types::*;
use ash::extensions::{ext, khr};
use ash::vk;
use cstr::cstr;
use inline_spirv::include_spirv;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr, CString};
use winit::window::Window;

const VALIDATION_LAYER: &CStr = cstr!("VK_LAYER_KHRONOS_validation");
const REQ_DEVICE_EXTENSIONS: [&CStr; 1] = [khr::Swapchain::name()];
const SWAPCHAIN_IMAGE_COUNT: u32 = 3;
const MAX_FRAMES_IN_FLIGHT: usize = 2;
type Vertex = ([f32; 2], [f32; 3]);

struct VulkanInstance {
    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils: Option<ext::DebugUtils>,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    surface_utils: khr::Surface,
}

impl VulkanInstance {
    fn new(window: &Window) -> VulkanResult<Self> {
        let entry = unsafe { ash::Entry::load()? };
        let validation_enabled = cfg!(debug_assertions) && Self::check_validation_support(&entry)?;
        let app_name = CString::new(env!("CARGO_PKG_NAME")).unwrap();
        let engine_name = cstr!("Snow3Derg");
        let instance = Self::create_instance(&entry, window, &app_name, engine_name, validation_enabled)?;
        let (debug_messenger, debug_utils) = if validation_enabled {
            let debug_utils = ext::DebugUtils::new(&entry, &instance);
            (Self::setup_debug_utils(&debug_utils)?, Some(debug_utils))
        } else {
            Default::default()
        };
        let surface_utils = khr::Surface::new(&entry, &instance);

        Ok(Self {
            entry,
            instance,
            debug_utils,
            debug_messenger,
            surface_utils,
        })
    }

    fn check_validation_support(entry: &ash::Entry) -> VulkanResult<bool> {
        let supported_layers = entry
            .enumerate_instance_layer_properties()
            .describe_err("Failed to enumerate instance layer properties")?;
        //eprintln!("Supported instance layers: {supported_layers:#?}");
        let validation_supp = supported_layers
            .iter()
            .any(|layer| vk_to_cstr(&layer.layer_name) == VALIDATION_LAYER);
        if !validation_supp {
            eprintln!("Validation layers requested but not available");
        }
        Ok(validation_supp)
    }

    fn check_portability_support(entry: &ash::Entry, names_ret: &mut Vec<*const c_char>) -> VulkanResult<bool> {
        let ext_list = entry
            .enumerate_instance_extension_properties(None)
            .describe_err("Failed to enumerate instance extension properties")?;
        let supported_exts: Vec<_> = ext_list.iter().map(|ext| vk_to_cstr(&ext.extension_name)).collect();
        //eprintln!("Supported instance extensions: {supported_exts:#?}");
        let ext_names = [vk::KhrPortabilityEnumerationFn::name(), khr::GetPhysicalDeviceProperties2::name()];
        let supported = ext_names.into_iter().all(|ext| supported_exts.contains(&ext));
        if supported {
            names_ret.extend(ext_names.map(CStr::as_ptr));
        }
        Ok(supported)
    }

    fn create_instance(
        entry: &ash::Entry, window: &Window, app_name: &CStr, engine_name: &CStr, validation_enabled: bool,
    ) -> VulkanResult<ash::Instance> {
        let mut extension_names = ash_window::enumerate_required_extensions(window.raw_display_handle())
            .describe_err("Unsupported display platform")?
            .to_vec();
        let portability = Self::check_portability_support(entry, &mut extension_names)?
            .then_some(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
            .unwrap_or_default();

        let mut layer_names = Vec::with_capacity(1);
        if validation_enabled {
            extension_names.push(ext::DebugUtils::name().as_ptr());
            layer_names.push(VALIDATION_LAYER.as_ptr());
            eprintln!("Using instance layer {VALIDATION_LAYER:?}");
        }

        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_0);

        let dbg_messenger_ci = create_debug_messenger_ci();
        let mut instance_ci = vk::InstanceCreateInfo::builder()
            .flags(portability)
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names);
        if validation_enabled {
            instance_ci.p_next = &dbg_messenger_ci as *const _ as _;
        }

        unsafe { entry.create_instance(&instance_ci, None).describe_err("Failed to create instance") }
    }

    fn setup_debug_utils(debug_utils: &ext::DebugUtils) -> VulkanResult<vk::DebugUtilsMessengerEXT> {
        let dbg_messenger_ci = create_debug_messenger_ci();
        let messenger = unsafe {
            debug_utils
                .create_debug_utils_messenger(&dbg_messenger_ci, None)
                .describe_err("Error creating debug utils callback")?
        };
        Ok(messenger)
    }

    fn create_surface(&self, window: &Window) -> VulkanResult<vk::SurfaceKHR> {
        unsafe {
            ash_window::create_surface(
                &self.entry,
                &self.instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
            .describe_err("Failed to create surface")
        }
    }

    fn pick_physical_device(&self, surface: vk::SurfaceKHR) -> VulkanResult<DeviceInfo> {
        let phys_devices = unsafe {
            self.instance
                .enumerate_physical_devices()
                .describe_err("Failed to enumerate physical devices")?
        };

        let dev_infos = phys_devices
            .into_iter()
            .map(|phys_dev| self.query_device_feature_support(phys_dev, surface))
            .filter(|result| !matches!(result, Err(VkError::UnsuitableDevice)))
            .collect::<Result<Vec<_>, _>>()?;

        dev_infos
            .into_iter()
            .max_by(|a, b| a.dev_type.cmp(&b.dev_type))
            .ok_or(VkError::EngineError("Failed to find a suitable GPU"))
    }

    fn query_device_feature_support(&self, phys_dev: vk::PhysicalDevice, surface: vk::SurfaceKHR) -> VulkanResult<DeviceInfo> {
        // device info
        let properties = unsafe { self.instance.get_physical_device_properties(phys_dev) };
        //let features = unsafe { self.instance.get_physical_device_features(phys_dev) };  //TODO: get limits and stuff
        let dev_type = properties.device_type.into();
        let name = vk_to_cstr(&properties.device_name).to_owned();

        // queue families
        let queue_families = unsafe { self.instance.get_physical_device_queue_family_properties(phys_dev) };
        let mut graphics_idx = None;
        let mut present_idx = None;
        for (que_family, idx) in queue_families.iter().zip(0..) {
            if que_family.queue_count == 0 {
                continue;
            }
            if que_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics_idx = Some(idx);
            }
            let present_supp = unsafe {
                self.surface_utils
                    .get_physical_device_surface_support(phys_dev, idx, surface)
                    .describe_err("Error querying surface support")?
            };
            if present_supp {
                present_idx = Some(idx);
            }
            if graphics_idx.is_some() && present_idx.is_some() {
                break;
            }
        }
        if graphics_idx.is_none() || present_idx.is_none() {
            eprintln!("Device {name:?} has incomplete queues");
            return Err(VkError::UnsuitableDevice);
        }

        // supported extensions
        let ext_list = unsafe {
            self.instance
                .enumerate_device_extension_properties(phys_dev)
                .describe_err("Failed to enumerate device extensions")?
        };
        let extensions: HashSet<_> = ext_list.iter().map(|ext| vk_to_cstr(&ext.extension_name).to_owned()).collect();
        //eprintln!("Supported device extensions: {extensions:#?}");
        let missing_ext = REQ_DEVICE_EXTENSIONS.into_iter().find(|&ext_name| !extensions.contains(ext_name));
        if let Some(ext_name) = missing_ext {
            eprintln!("Device {name:?} has missing extension {ext_name:?}");
            return Err(VkError::UnsuitableDevice);
        }

        // memory properties
        let mem_props = unsafe { self.instance.get_physical_device_memory_properties(phys_dev) };

        Ok(DeviceInfo {
            phys_dev,
            dev_type,
            name,
            graphics_idx: graphics_idx.unwrap(),
            present_idx: present_idx.unwrap(),
            extensions,
            mem_props,
        })
    }

    fn query_surface_info(&self, phys_dev: vk::PhysicalDevice, surface: vk::SurfaceKHR) -> VulkanResult<SurfaceInfo> {
        let surf_caps = unsafe {
            self.surface_utils
                .get_physical_device_surface_capabilities(phys_dev, surface)
                .describe_err("Failed to get surface capabilities")?
        };
        let surf_formats = unsafe {
            self.surface_utils
                .get_physical_device_surface_formats(phys_dev, surface)
                .describe_err("Failed to get surface formats")?
        };
        let present_modes = unsafe {
            self.surface_utils
                .get_physical_device_surface_present_modes(phys_dev, surface)
                .describe_err("Failed to get surface present modes")?
        };
        if surf_formats.is_empty() || present_modes.is_empty() {
            return Err(VkError::EngineError("Device has incomplete surface capabilities"));
        }

        Ok(SurfaceInfo {
            capabilities: surf_caps,
            formats: surf_formats,
            present_modes,
        })
    }

    fn create_logical_device(&self, dev_info: &DeviceInfo) -> VulkanResult<ash::Device> {
        let queue_prio = [1.0];
        let queues_ci: Vec<_> = dev_info
            .unique_families()
            .into_iter()
            .map(|idx| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(idx)
                    .queue_priorities(&queue_prio)
                    .build()
            })
            .collect();

        let features = vk::PhysicalDeviceFeatures::default();
        let mut extensions = REQ_DEVICE_EXTENSIONS.map(CStr::as_ptr).to_vec();
        if dev_info.extensions.contains(vk::KhrPortabilitySubsetFn::name()) {
            extensions.push(vk::KhrPortabilitySubsetFn::name().as_ptr());
        }

        let device_ci = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queues_ci)
            .enabled_features(&features)
            .enabled_extension_names(&extensions);

        unsafe {
            self.instance
                .create_device(dev_info.phys_dev, &device_ci, None)
                .describe_err("Failed to create logical device")
        }
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            if let Some(debug_utils) = &self.debug_utils {
                debug_utils.destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Debug, Default)]
struct DeviceInfo {
    phys_dev: vk::PhysicalDevice,
    name: CString,
    dev_type: DeviceType,
    graphics_idx: u32,
    present_idx: u32,
    extensions: HashSet<CString>,
    mem_props: vk::PhysicalDeviceMemoryProperties,
}

impl DeviceInfo {
    fn unique_families(&self) -> Vec<u32> {
        if self.graphics_idx == self.present_idx {
            vec![self.graphics_idx]
        } else {
            vec![self.graphics_idx, self.present_idx]
        }
    }

    fn find_memory_type(&self, type_filter: u32, prop_flags: vk::MemoryPropertyFlags) -> VulkanResult<u32> {
        for i in 0..self.mem_props.memory_type_count {
            if type_filter & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(prop_flags) {
                return Ok(i);
            }
        }
        Err(VkError::EngineError("Failed to find suitable memory type"))
    }
}

#[derive(Debug)]
struct SurfaceInfo {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SurfaceInfo {
    fn surface_format(&self) -> &vk::SurfaceFormatKHR {
        self.formats
            .iter()
            .find(|&fmt| fmt.format == vk::Format::B8G8R8A8_SRGB && fmt.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .or_else(|| self.formats.first())
            .expect("Empty surface formats")
    }

    fn present_mode(&self) -> vk::PresentModeKHR {
        self.present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::IMMEDIATE)
            .unwrap_or(vk::PresentModeKHR::FIFO)
    }

    fn calc_extent(&self, width: u32, height: u32) -> vk::Extent2D {
        if self.capabilities.current_extent.width != u32::max_value() {
            self.capabilities.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(self.capabilities.min_image_extent.width, self.capabilities.max_image_extent.width),
                height: height.clamp(self.capabilities.min_image_extent.height, self.capabilities.max_image_extent.height),
            }
        }
    }

    fn calc_image_count(&self, count: u32) -> u32 {
        if self.capabilities.max_image_count > 0 {
            count.clamp(self.capabilities.min_image_count, self.capabilities.max_image_count)
        } else {
            count.max(self.capabilities.min_image_count)
        }
    }
}

#[derive(Debug)]
struct SwapchainInfo {
    handle: vk::SwapchainKHR,
    //images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    format: vk::Format,
    extent: vk::Extent2D,
}

impl SwapchainInfo {
    fn viewport(&self) -> vk::Viewport {
        vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.extent.width as f32,
            height: self.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    fn extent_rect(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: Default::default(),
            extent: self.extent,
        }
    }

    unsafe fn cleanup(&mut self, device: &VulkanDevice) {
        for &imgview in &self.image_views {
            device.destroy_image_view(imgview, None);
        }
        device.swapchain_utils.destroy_swapchain(self.handle, None);
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
enum DeviceType {
    #[default]
    Other,
    Cpu,
    IntegratedGpu,
    VirtualGpu,
    DiscreteGpu,
}

impl From<vk::PhysicalDeviceType> for DeviceType {
    fn from(dev_type: vk::PhysicalDeviceType) -> Self {
        match dev_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::DiscreteGpu,
            vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceType::VirtualGpu,
            vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::IntegratedGpu,
            vk::PhysicalDeviceType::CPU => DeviceType::Cpu,
            _ => DeviceType::Other,
        }
    }
}

pub struct VulkanDevice {
    instance: VulkanInstance,
    surface: vk::SurfaceKHR,
    dev_info: DeviceInfo,
    surface_info: SurfaceInfo,
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_utils: khr::Swapchain,
}

impl VulkanDevice {
    pub fn new(window: &Window) -> VulkanResult<Self> {
        let vk = VulkanInstance::new(window)?;
        let surface = vk.create_surface(window)?;
        let dev_info = vk.pick_physical_device(surface)?;
        eprintln!("Selected device: {:?}", dev_info.name);
        let surface_info = vk.query_surface_info(dev_info.phys_dev, surface)?;
        let device = vk.create_logical_device(&dev_info)?;
        let graphics_queue = unsafe { device.get_device_queue(dev_info.graphics_idx, 0) };
        let present_queue = unsafe { device.get_device_queue(dev_info.present_idx, 0) };
        let swapchain_utils = khr::Swapchain::new(&vk.instance, &device);

        Ok(Self {
            instance: vk,
            surface,
            dev_info,
            surface_info,
            device,
            graphics_queue,
            present_queue,
            swapchain_utils,
        })
    }

    fn create_swapchain(&self, window: &Window, image_count: u32, old_swapchain: Option<vk::SwapchainKHR>) -> VulkanResult<SwapchainInfo> {
        let surface_format = self.surface_info.surface_format();
        let present_mode = self.surface_info.present_mode();
        let win_size = window.inner_size();
        eprintln!("window size: {} x {}", win_size.width, win_size.height);
        let extent = self.surface_info.calc_extent(win_size.width, win_size.height);
        let img_count = self.surface_info.calc_image_count(image_count);
        let que_families = self.dev_info.unique_families();
        let img_sharing_mode = if que_families.len() > 1 {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let swapchain_ci = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface)
            .min_image_count(img_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(img_sharing_mode)
            .queue_family_indices(&que_families)
            .pre_transform(self.surface_info.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain.unwrap_or_default());

        let swapchain = unsafe {
            self.swapchain_utils
                .create_swapchain(&swapchain_ci, None)
                .describe_err("Failed to create swapchain")?
        };

        let images = unsafe {
            self.swapchain_utils
                .get_swapchain_images(swapchain)
                .describe_err("Failed to get swapchain images")?
        };

        let image_views = images
            .iter()
            .map(|&image| {
                let imageview_ci = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);

                unsafe { self.device.create_image_view(&imageview_ci, None) }
            })
            .collect::<Result<_, _>>()
            .describe_err("Failed to create image views")?;

        Ok(SwapchainInfo {
            handle: swapchain,
            //images,
            image_views,
            format: surface_format.format,
            extent,
        })
    }

    fn create_shader_module(&self, spirv_code: &[u32]) -> VulkanResult<vk::ShaderModule> {
        let shader_ci = vk::ShaderModuleCreateInfo::builder().code(spirv_code);

        unsafe {
            self.device
                .create_shader_module(&shader_ci, None)
                .describe_err("Failed to create shader module")
        }
    }

    fn create_render_pass(&self, format: vk::Format) -> VulkanResult<vk::RenderPass> {
        let color_attach = [vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        }];

        let attach_ref = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let subpass = [vk::SubpassDescription::builder().color_attachments(&attach_ref).build()];

        let dependency = [vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .build()];

        let render_pass_ci = vk::RenderPassCreateInfo::builder()
            .attachments(&color_attach)
            .subpasses(&subpass)
            .dependencies(&dependency);

        unsafe {
            self.device
                .create_render_pass(&render_pass_ci, None)
                .describe_err("Failed to create render pass")
        }
    }

    fn create_graphics_pipeline(
        &self, vert_shader_spv: &[u32], frag_shader_spv: &[u32], render_pass: vk::RenderPass,
    ) -> VulkanResult<(vk::Pipeline, vk::PipelineLayout)> {
        let vert_shader = self.create_shader_module(vert_shader_spv)?;
        let frag_shader = self.create_shader_module(frag_shader_spv)?;

        let entry_point = cstr!("main");
        let shader_stages_ci = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader)
                .name(entry_point)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader)
                .name(entry_point)
                .build(),
        ];

        let binding_desc = [Vertex::binding_desc(0)];
        let attr_desc = Vertex::attr_desc(0);

        let vertex_input_ci = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_desc)
            .vertex_attribute_descriptions(&attr_desc);

        let input_assembly_ci = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
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
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE);

        let multisample_ci = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0);

        let color_attach = [vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .build()];

        let color_blend_ci = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_attach);

        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_ci, None)
                .describe_err("Failed to create pipeline layout")?
        };

        let pipeline_ci = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages_ci)
            .vertex_input_state(&vertex_input_ci)
            .input_assembly_state(&input_assembly_ci)
            .viewport_state(&viewport_state_ci)
            .rasterization_state(&rasterizer_ci)
            .multisample_state(&multisample_ci)
            .color_blend_state(&color_blend_ci)
            .dynamic_state(&dynamic_state_ci)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .build();

        let pipeline = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
                .map_err(|(_, err)| VkError::VulkanMsg("Error creating pipeline", err))?
        };

        unsafe {
            self.device.destroy_shader_module(vert_shader, None);
            self.device.destroy_shader_module(frag_shader, None);
        }

        Ok((pipeline[0], pipeline_layout))
    }

    fn create_framebuffers(&self, swapchain: &SwapchainInfo, render_pass: vk::RenderPass) -> VulkanResult<Vec<vk::Framebuffer>> {
        swapchain
            .image_views
            .iter()
            .map(|&imgview| {
                let attachments = [imgview];
                let framebuffer_ci = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .layers(1);

                unsafe {
                    self.device
                        .create_framebuffer(&framebuffer_ci, None)
                        .describe_err("Failed to create framebuffer")
                }
            })
            .collect()
    }

    fn create_command_pool(&self, flags: vk::CommandPoolCreateFlags) -> VulkanResult<vk::CommandPool> {
        let command_pool_ci = vk::CommandPoolCreateInfo::builder()
            .flags(flags)
            .queue_family_index(self.dev_info.graphics_idx);

        unsafe {
            self.device
                .create_command_pool(&command_pool_ci, None)
                .describe_err("Failed to create command pool")
        }
    }

    fn create_command_buffers(&self, pool: vk::CommandPool, count: u32) -> VulkanResult<Vec<vk::CommandBuffer>> {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .describe_err("Failed to allocate command buffers")
        }
    }

    fn create_semaphore(&self) -> VulkanResult<vk::Semaphore> {
        let semaphore_ci = vk::SemaphoreCreateInfo::default();
        unsafe {
            self.device
                .create_semaphore(&semaphore_ci, None)
                .describe_err("Failed to create semaphore")
        }
    }

    fn create_fence(&self) -> VulkanResult<vk::Fence> {
        let fence_ci = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        unsafe { self.device.create_fence(&fence_ci, None).describe_err("Failed to create fence") }
    }

    fn wait_idle(&self) -> VulkanResult<()> {
        unsafe { self.device.device_wait_idle().describe_err("Failed to wait device idle") }
    }

    fn update_surface_info(&mut self) -> VulkanResult<()> {
        self.surface_info = self.instance.query_surface_info(self.dev_info.phys_dev, self.surface)?;
        Ok(())
    }

    fn allocate_buffer(
        &self, size: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags,
    ) -> VulkanResult<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_ci = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_ci, None)
                .describe_err("Failed to create buffer")?
        };

        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mem_type = self.dev_info.find_memory_type(mem_reqs.memory_type_bits, properties)?;

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type);

        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .describe_err("Failed to allocate buffer memory")?
        };
        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .describe_err("Failed to bind buffer memory")?
        };

        Ok((buffer, memory))
    }

    fn write_memory<T: Copy>(&self, memory: vk::DeviceMemory, data: &[T]) -> VulkanResult<()> {
        let size = std::mem::size_of_val(data) as _;
        unsafe {
            let mapped_ptr = self
                .device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                .describe_err("Failed to map buffer memory")? as *mut T;
            std::slice::from_raw_parts_mut(mapped_ptr, data.len()).copy_from_slice(data);
            self.device.unmap_memory(memory);
        };
        Ok(())
    }

    fn copy_buffer(&self, dst_buffer: vk::Buffer, src_buffer: vk::Buffer, size: vk::DeviceSize, pool: vk::CommandPool) -> VulkanResult<()> {
        let cmd_buffer = self.create_command_buffers(pool, 1)?;
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer[0], &begin_info)
                .describe_err("Failed to begin recording command buffer")?;
            self.device.cmd_copy_buffer(cmd_buffer[0], src_buffer, dst_buffer, &[copy_region]);
            self.device
                .end_command_buffer(cmd_buffer[0])
                .describe_err("Failed to end recording command buffer")?;
        }

        let submit_info = [vk::SubmitInfo::builder().command_buffers(&cmd_buffer).build()];
        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &submit_info, vk::Fence::null())
                .describe_err("Failed to submit queue")?;
            self.device
                .queue_wait_idle(self.graphics_queue)
                .describe_err("Failed to wait queue idle")?;
            self.device.free_command_buffers(pool, &cmd_buffer);
        }

        Ok(())
    }

    fn create_buffer<T: Copy>(
        &self, data: &[T], usage: vk::BufferUsageFlags, pool: vk::CommandPool,
    ) -> VulkanResult<(vk::Buffer, vk::DeviceMemory)> {
        let size = std::mem::size_of_val(data) as _;
        let (src_buffer, src_memory) = self.allocate_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let (dst_buffer, dst_memory) = self.allocate_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        self.write_memory(src_memory, data)?;
        self.copy_buffer(dst_buffer, src_buffer, size, pool)?;

        unsafe {
            self.device.destroy_buffer(src_buffer, None);
            self.device.free_memory(src_memory, None);
        }

        Ok((dst_buffer, dst_memory))
    }
}

impl std::ops::Deref for VulkanDevice {
    type Target = ash::Device;
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.instance.surface_utils.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
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

fn vk_to_cstr(raw: &[c_char]) -> &CStr {
    //TODO: replace with `CStr::from_bytes_until_nul` when it's stable
    unsafe { CStr::from_ptr(raw.as_ptr()) }
}

extern "system" fn vulkan_debug_utils_callback(
    msg_severity: vk::DebugUtilsMessageSeverityFlagsEXT, msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    cb_data: *const vk::DebugUtilsMessengerCallbackDataEXT, _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*cb_data).p_message) }.to_string_lossy();
    eprintln!("Debug: [{msg_severity:?}][{msg_type:?}] {message}");
    vk::FALSE
}

fn create_debug_messenger_ci() -> vk::DebugUtilsMessengerCreateInfoEXT {
    vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
            //| vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            //| vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(vulkan_debug_utils_callback))
        .build()
}
