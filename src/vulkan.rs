use ash::extensions::{ext, khr};
use ash::vk;
use cstr::cstr;
use inline_spirv::include_spirv;
use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr, CString};
use winit::dpi::PhysicalSize;
use winit::window::Window;

const VALIDATION_LAYER: &CStr = cstr!("VK_LAYER_KHRONOS_validation");
const REQ_INSTANCE_EXTENSIONS: [&CStr; 2] = [
    khr::Surface::name(),
    #[cfg(target_family = "unix")]
    khr::XlibSurface::name(),
    #[cfg(target_family = "windows")]
    khr::Win32Surface::name(),
];
const REQ_DEVICE_EXTENSIONS: [&CStr; 1] = [khr::Swapchain::name()];
const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct VulkanInstance {
    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils: Option<ext::DebugUtils>,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    surface_utils: khr::Surface,
}

impl VulkanInstance {
    fn new() -> VulkanResult<Self> {
        let entry = unsafe { ash::Entry::load().map_err(Error::LoadingError)? };
        let validation_enabled = cfg!(debug_assertions) && Self::check_validation_support(&entry)?;
        let app_name = CString::new(env!("CARGO_PKG_NAME")).unwrap();
        let engine_name = cstr!("Snow3Derg");
        let instance = Self::create_instance(&entry, &app_name, &engine_name, validation_enabled)?;
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
            .map_err(Error::bind_msg("Failed to enumerate instance layer properties"))?;
        //eprintln!("Supported instance layers: {supported_layers:#?}");
        let validation_supp = supported_layers
            .iter()
            .any(|layer| vk_to_cstr(&layer.layer_name) == VALIDATION_LAYER);
        if !validation_supp {
            eprintln!("Validation layers requested but not available");
        }
        Ok(validation_supp)
    }

    fn get_required_instance_extensions(entry: &ash::Entry, validation_enabled: bool) -> VulkanResult<Vec<*const i8>> {
        let ext_list = entry
            .enumerate_instance_extension_properties(None)
            .map_err(Error::bind_msg("Failed to enumerate instance extension properties"))?;
        let supported_exts: Vec<_> = ext_list.iter().map(|ext| vk_to_cstr(&ext.extension_name)).collect();
        //eprintln!("Supported instance extensions: {supported_exts:#?}");
        REQ_INSTANCE_EXTENSIONS
            .iter()
            .cloned()
            .chain(validation_enabled.then(ext::DebugUtils::name))
            .map(|ext| {
                if supported_exts.contains(&ext) {
                    eprintln!("Using instance extension {ext:?}");
                    Ok(ext.as_ptr())
                } else {
                    eprintln!("Unsupported instance extension {ext:?}");
                    Err(Error::EngineError("Couldn't find all required instance extensions"))
                }
            })
            .collect()
    }

    fn create_instance(entry: &ash::Entry, app_name: &CStr, engine_name: &CStr, validation_enabled: bool) -> VulkanResult<ash::Instance> {
        let extension_names = Self::get_required_instance_extensions(entry, validation_enabled)?;

        let mut layer_names = Vec::with_capacity(1);
        if validation_enabled {
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
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names);
        if validation_enabled {
            instance_ci.p_next = &dbg_messenger_ci as *const _ as _;
        }

        unsafe {
            entry
                .create_instance(&instance_ci, None)
                .map_err(Error::bind_msg("Failed to create instance"))
        }
    }

    fn setup_debug_utils(debug_utils: &ext::DebugUtils) -> VulkanResult<vk::DebugUtilsMessengerEXT> {
        let dbg_messenger_ci = create_debug_messenger_ci();
        let messenger = unsafe {
            debug_utils
                .create_debug_utils_messenger(&dbg_messenger_ci, None)
                .map_err(Error::bind_msg("Error creating debug utils callback"))?
        };
        Ok(messenger)
    }

    #[cfg(target_family = "unix")]
    fn create_surface(&self, window: &Window) -> VulkanResult<vk::SurfaceKHR> {
        use winit::platform::unix::WindowExtUnix;

        let surface_ci = vk::XlibSurfaceCreateInfoKHR::builder()
            .window(window.xlib_window().ok_or(Error::EngineError("Failed to get Xlib window"))?)
            .dpy(window.xlib_display().ok_or(Error::EngineError("Failed to get Xlib display"))? as _);
        let surface_loader = khr::XlibSurface::new(&self.entry, &self.instance);
        unsafe {
            surface_loader
                .create_xlib_surface(&surface_ci, None)
                .map_err(Error::bind_msg("Failed to create Xlib surface"))
        }
    }

    #[cfg(target_family = "windows")]
    fn create_surface(&self, window: &Window) -> VulkanResult<vk::SurfaceKHR> {
        use winit::platform::windows::WindowExtWindows;

        let surface_ci = vk::Win32SurfaceCreateInfoKHR::builder()
            .hwnd(window.hwnd() as _)
            .hinstance(window.hinstance() as _);
        let surface_loader = khr::Win32Surface::new(&self.entry, &self.instance);
        unsafe {
            surface_loader
                .create_win32_surface(&surface_ci, None)
                .map_err(Error::bind_msg("Failed to create Win32 surface"))
        }
    }

    fn pick_physical_device(&self, surface: vk::SurfaceKHR) -> VulkanResult<DeviceInfo> {
        let phys_devices = unsafe {
            self.instance
                .enumerate_physical_devices()
                .map_err(Error::bind_msg("Failed to enumerate physical devices"))?
        };

        let dev_infos = phys_devices
            .into_iter()
            .map(|phys_dev| self.query_device_feature_support(phys_dev, surface))
            .filter(|result| !matches!(result, Err(Error::UnsuitableDevice)))
            .collect::<Result<Vec<_>, _>>()?;

        dev_infos
            .into_iter()
            .max_by(|a, b| a.dev_type.cmp(&b.dev_type))
            .ok_or(Error::EngineError("Failed to find a suitable GPU"))
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
                    .map_err(Error::bind_msg("Error querying surface support"))?
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
            return Err(Error::UnsuitableDevice);
        }

        // supported extensions
        let ext_list = unsafe {
            self.instance
                .enumerate_device_extension_properties(phys_dev)
                .map_err(Error::bind_msg("Failed to enumerate device extensions"))?
        };
        let extensions: HashSet<_> = ext_list.iter().map(|ext| vk_to_cstr(&ext.extension_name).to_owned()).collect();
        //eprintln!("Supported device extensions: {extensions:#?}");
        let missing_ext = REQ_DEVICE_EXTENSIONS.into_iter().find(|&ext_name| !extensions.contains(ext_name));
        if let Some(ext_name) = missing_ext {
            eprintln!("Device {name:?} has missing extension {ext_name:?}");
            return Err(Error::UnsuitableDevice);
        }

        // swapchain support
        let surf_caps = unsafe {
            self.surface_utils
                .get_physical_device_surface_capabilities(phys_dev, surface)
                .map_err(Error::bind_msg("Failed to get surface capabilities"))?
        };
        let surf_formats = unsafe {
            self.surface_utils
                .get_physical_device_surface_formats(phys_dev, surface)
                .map_err(Error::bind_msg("Failed to get surface formats"))?
        };
        let present_modes = unsafe {
            self.surface_utils
                .get_physical_device_surface_present_modes(phys_dev, surface)
                .map_err(Error::bind_msg("Failed to get surface present modes"))?
        };
        if surf_formats.is_empty() || present_modes.is_empty() {
            eprintln!("Device {name:?} has incomplete surface capabilities");
            return Err(Error::UnsuitableDevice);
        }

        Ok(DeviceInfo {
            phys_dev,
            dev_type,
            name,
            graphics_idx: graphics_idx.unwrap(),
            present_idx: present_idx.unwrap(),
            //extensions,
            surf_caps,
            surf_formats,
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
        let layers = [VALIDATION_LAYER.as_ptr()];
        let extensions: Vec<_> = REQ_DEVICE_EXTENSIONS.into_iter().map(CStr::as_ptr).collect();

        let mut device_ci = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queues_ci)
            .enabled_features(&features)
            .enabled_extension_names(&extensions);

        if self.debug_utils.is_some() {
            // validation enabled
            device_ci = device_ci.enabled_layer_names(&layers);
        }

        unsafe {
            self.instance
                .create_device(dev_info.phys_dev, &device_ci, None)
                .map_err(Error::bind_msg("Failed to create logical device"))
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

#[derive(Debug, Default)]
struct DeviceInfo {
    phys_dev: vk::PhysicalDevice,
    name: CString,
    dev_type: DeviceType,
    graphics_idx: u32,
    present_idx: u32,
    //extensions: HashSet<CString>,
    surf_caps: vk::SurfaceCapabilitiesKHR,
    surf_formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl DeviceInfo {
    fn unique_families(&self) -> Vec<u32> {
        if self.graphics_idx == self.present_idx {
            vec![self.graphics_idx]
        } else {
            vec![self.graphics_idx, self.present_idx]
        }
    }
}

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
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_utils: khr::Swapchain,
}

impl VulkanDevice {
    pub fn new(window: &Window) -> VulkanResult<Self> {
        let vk = VulkanInstance::new()?;
        let surface = vk.create_surface(&window)?;
        let dev_info = vk.pick_physical_device(surface)?;
        eprintln!("Selected device: {:?}", dev_info.name);
        let device = vk.create_logical_device(&dev_info)?;
        let graphics_queue = unsafe { device.get_device_queue(dev_info.graphics_idx, 0) };
        let present_queue = unsafe { device.get_device_queue(dev_info.present_idx, 0) };
        let swapchain_utils = khr::Swapchain::new(&vk.instance, &device);

        Ok(Self {
            instance: vk,
            surface,
            dev_info,
            device,
            graphics_queue,
            present_queue,
            swapchain_utils,
        })
    }

    fn create_swapchain(&self, window_width: u32, window_height: u32, image_count: u32) -> VulkanResult<SwapchainInfo> {
        let surface_format = self
            .dev_info
            .surf_formats
            .iter()
            .find(|&fmt| fmt.format == vk::Format::B8G8R8A8_SRGB && fmt.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .or_else(|| self.dev_info.surf_formats.first())
            .ok_or(Error::EngineError("Empty surface formats"))?;

        let present_mode = self
            .dev_info
            .present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::IMMEDIATE)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let extent = if self.dev_info.surf_caps.current_extent.width != u32::max_value() {
            self.dev_info.surf_caps.current_extent
        } else {
            vk::Extent2D {
                width: window_width.clamp(
                    self.dev_info.surf_caps.min_image_extent.width,
                    self.dev_info.surf_caps.max_image_extent.width,
                ),
                height: window_height.clamp(
                    self.dev_info.surf_caps.min_image_extent.height,
                    self.dev_info.surf_caps.max_image_extent.height,
                ),
            }
        };

        let img_count = if self.dev_info.surf_caps.max_image_count > 0 {
            image_count.clamp(self.dev_info.surf_caps.min_image_count, self.dev_info.surf_caps.max_image_count)
        } else {
            image_count.max(self.dev_info.surf_caps.min_image_count)
        };

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
            .pre_transform(self.dev_info.surf_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe {
            self.swapchain_utils
                .create_swapchain(&swapchain_ci, None)
                .map_err(Error::bind_msg("Failed to create swapchain"))?
        };

        let images = unsafe {
            self.swapchain_utils
                .get_swapchain_images(swapchain)
                .map_err(Error::bind_msg("Failed to get swapchain images"))?
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
            .map_err(Error::bind_msg("Failed to create image views"))?;

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
                .map_err(Error::bind_msg("Failed to create shader module"))
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
                .map_err(Error::bind_msg("Failed to create render pass"))
        }
    }

    fn create_graphics_pipeline(
        &self, render_pass: vk::RenderPass, swapchain: &SwapchainInfo,
    ) -> VulkanResult<(vk::Pipeline, vk::PipelineLayout)> {
        let vert_shader = self.create_shader_module(include_spirv!("src/shaders/triangle.vert.glsl", vert, glsl))?;
        let frag_shader = self.create_shader_module(include_spirv!("src/shaders/triangle.frag.glsl", frag, glsl))?;

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

        let vertex_input_ci = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly_ci = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = [swapchain.viewport()];
        let scissor = [swapchain.extent_rect()];

        let viewport_state_ci = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewport)
            .scissors(&scissor);

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
            .build()];

        let color_blend_ci = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_attach);

        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_ci, None)
                .map_err(Error::bind_msg("Failed to create pipeline layout"))?
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
                .map_err(|(_, err)| Error::VulkanError("Error creating pipeline", err))?
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
                        .map_err(Error::bind_msg("Failed to create framebuffer"))
                }
            })
            .collect()
    }

    fn create_command_pool(&self) -> VulkanResult<vk::CommandPool> {
        let command_pool_ci = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.dev_info.graphics_idx);

        unsafe {
            self.device
                .create_command_pool(&command_pool_ci, None)
                .map_err(Error::bind_msg("Failed to create command pool"))
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
                .map_err(Error::bind_msg("Failed to allocate command buffers"))
        }
    }

    fn create_semaphore(&self) -> VulkanResult<vk::Semaphore> {
        let semaphore_ci = vk::SemaphoreCreateInfo::default();
        unsafe {
            self.device
                .create_semaphore(&semaphore_ci, None)
                .map_err(Error::bind_msg("Failed to create semaphore"))
        }
    }

    fn create_fence(&self) -> VulkanResult<vk::Fence> {
        let fence_ci = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        unsafe {
            self.device
                .create_fence(&fence_ci, None)
                .map_err(Error::bind_msg("Failed to create fence"))
        }
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

pub struct VulkanApp {
    device: VulkanDevice,
    swapchain: SwapchainInfo,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_avail_sems: Vec<vk::Semaphore>,
    render_finished_sems: Vec<vk::Semaphore>,
    in_flight_fens: Vec<vk::Fence>,
    current_frame: usize,
}

impl VulkanApp {
    pub fn new(window: &Window) -> VulkanResult<Self> {
        let vk = VulkanDevice::new(window)?;
        let PhysicalSize { width, height } = window.inner_size();
        eprintln!("window size: {width} x {height}");
        let swapchain = vk.create_swapchain(width, height, 3)?;
        let render_pass = vk.create_render_pass(swapchain.format)?;
        let (pipeline, pipeline_layout) = vk.create_graphics_pipeline(render_pass, &swapchain)?;
        let framebuffers = vk.create_framebuffers(&swapchain, render_pass)?;
        let command_pool = vk.create_command_pool()?;
        let command_buffers = vk.create_command_buffers(command_pool, MAX_FRAMES_IN_FLIGHT as u32)?;
        let image_avail_sems = (0..MAX_FRAMES_IN_FLIGHT).map(|_| vk.create_semaphore()).collect::<Result<_, _>>()?;
        let render_finished_sems = (0..MAX_FRAMES_IN_FLIGHT).map(|_| vk.create_semaphore()).collect::<Result<_, _>>()?;
        let in_flight_fens = (0..MAX_FRAMES_IN_FLIGHT).map(|_| vk.create_fence()).collect::<Result<_, _>>()?;

        Ok(Self {
            device: vk,
            swapchain,
            render_pass,
            pipeline,
            pipeline_layout,
            framebuffers,
            command_pool,
            command_buffers,
            image_avail_sems,
            render_finished_sems,
            in_flight_fens,
            current_frame: 0,
        })
    }

    fn record_command_buffer(&self, cmd_buffer: vk::CommandBuffer, image_idx: u32) -> VulkanResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .map_err(Error::bind_msg("Failed to begin recording command buffer"))?;
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
        unsafe {
            self.device
                .cmd_begin_render_pass(cmd_buffer, &renderpass_info, vk::SubpassContents::INLINE);
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_set_viewport(cmd_buffer, 0, &viewport);
            self.device.cmd_set_scissor(cmd_buffer, 0, &scissor);
            self.device.cmd_draw(cmd_buffer, 3, 1, 0, 0);
            self.device.cmd_end_render_pass(cmd_buffer);
            self.device
                .end_command_buffer(cmd_buffer)
                .map_err(Error::bind_msg("Failed to record command buffer"))?;
        }

        Ok(())
    }

    pub fn draw_frame(&mut self) -> VulkanResult<bool> {
        let in_flight_fen = [self.in_flight_fens[self.current_frame]];
        let image_avail_sem = [self.image_avail_sems[self.current_frame]];
        let render_finish_sem = [self.render_finished_sems[self.current_frame]];
        let command_buffer = [self.command_buffers[self.current_frame]];
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        let image_indices = unsafe {
            self.device
                .wait_for_fences(&in_flight_fen, true, u64::MAX)
                .map_err(Error::bind_msg("Failed waiting for error"))?;
            self.device
                .reset_fences(&in_flight_fen)
                .map_err(Error::bind_msg("Failed resetting fences"))?;
            let (image_idx, _) = self
                .device
                .swapchain_utils
                .acquire_next_image(self.swapchain.handle, u64::MAX, image_avail_sem[0], vk::Fence::null())
                .map_err(Error::bind_msg("Failed to acquire swapchain image"))?;
            self.device
                .reset_command_buffer(command_buffer[0], vk::CommandBufferResetFlags::empty())
                .map_err(Error::bind_msg("Failed to reset command buffer"))?;

            self.record_command_buffer(command_buffer[0], image_idx)?;

            [image_idx]
        };

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
                .map_err(Error::bind_msg("Failed to submit draw command buffer"))?
        }

        let swapchains = [self.swapchain.handle];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&render_finish_sem)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.device
                .swapchain_utils
                .queue_present(self.device.present_queue, &present_info)
                .map_err(Error::bind_msg("Failed to present queue"))
        }
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().expect("Failed to wait idle");
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device.destroy_semaphore(self.image_avail_sems[i], None);
                self.device.destroy_semaphore(self.render_finished_sems[i], None);
                self.device.destroy_fence(self.in_flight_fens[i], None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            for &imgview in &self.swapchain.image_views {
                self.device.destroy_image_view(imgview, None);
            }
            self.device.swapchain_utils.destroy_swapchain(self.swapchain.handle, None);
        }
    }
}

pub type VulkanResult<T> = Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    LoadingError(ash::LoadingError),
    VulkanError(&'static str, vk::Result),
    EngineError(&'static str),
    UnsuitableDevice, // used internally
}

impl Error {
    const fn bind_msg(msg: &'static str) -> impl Fn(vk::Result) -> Self {
        move |err| Self::VulkanError(msg, err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Self::LoadingError(err) => write!(f, "Failed to load Vulkan library: {err}"),
            Self::VulkanError(desc, err) => write!(f, "{desc}: {err}"),
            Self::EngineError(desc) => write!(f, "{desc}"),
            Self::UnsuitableDevice => write!(f, "Unsuitable device"),
        }
    }
}

fn vk_to_cstr(raw: &[c_char]) -> &CStr {
    //TODO: replace with `CStr::from_bytes_until_nul` when it's stable
    unsafe { CStr::from_ptr(raw.as_ptr()) }
}
