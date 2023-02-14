use crate::types::*;
use ash::extensions::{ext, khr};
use ash::vk;
use cstr::cstr;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::collections::HashSet;
use std::convert::identity;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Arc;

const VALIDATION_LAYER: &CStr = cstr!("VK_LAYER_KHRONOS_validation");
const REQ_DEVICE_EXTENSIONS: [&CStr; 1] = [khr::Swapchain::name()];

pub struct VulkanInstance {
    entry: ash::Entry,
    instance: ash::Instance,
    pub debug_utils: Option<ext::DebugUtils>,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    pub surface_utils: khr::Surface,
}

impl VulkanInstance {
    pub fn new<W: HasRawDisplayHandle>(window: &W, app_name: &str) -> VulkanResult<Arc<Self>> {
        let entry = unsafe { ash::Entry::load()? };
        let validation_enabled = cfg!(debug_assertions) && Self::check_validation_support(&entry)?;
        let app_name = CString::new(app_name).unwrap();
        let engine_name = CString::new(env!("CARGO_PKG_NAME")).unwrap();
        let instance = Self::create_instance(&entry, window, &app_name, &engine_name, validation_enabled)?;
        let (debug_messenger, debug_utils) = if validation_enabled {
            let debug_utils = ext::DebugUtils::new(&entry, &instance);
            (Self::setup_debug_utils(&debug_utils)?, Some(debug_utils))
        } else {
            Default::default()
        };
        let surface_utils = khr::Surface::new(&entry, &instance);

        Ok(Arc::new(Self {
            entry,
            instance,
            debug_utils,
            debug_messenger,
            surface_utils,
        }))
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

    fn create_instance<W: HasRawDisplayHandle>(
        entry: &ash::Entry, window: &W, app_name: &CStr, engine_name: &CStr, validation_enabled: bool,
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

    pub fn create_surface<W>(&self, window: &W) -> VulkanResult<vk::SurfaceKHR>
    where
        W: HasRawDisplayHandle + HasRawWindowHandle,
    {
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

    pub fn pick_physical_device(&self, surface: vk::SurfaceKHR, selection: DeviceSelection) -> VulkanResult<DeviceInfo> {
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

        if selection.is_empty() {
            dev_infos
                .into_iter()
                .max_by(|a, b| a.dev_type.cmp(&b.dev_type))
                .ok_or(VkError::EngineError("Failed to find a suitable GPU"))
        } else {
            dev_infos
                .into_iter()
                .find(|dev| selection.matches(dev))
                .ok_or(VkError::EngineError("Failed to find a suitable GPU matching the selection"))
        }
    }

    fn query_device_feature_support(&self, phys_dev: vk::PhysicalDevice, surface: vk::SurfaceKHR) -> VulkanResult<DeviceInfo> {
        // device info
        let properties = unsafe { self.instance.get_physical_device_properties(phys_dev) };
        //let features = unsafe { self.instance.get_physical_device_features(phys_dev) };  //TODO: get limits and stuff
        let dev_type = properties.device_type.into();
        let name = vk_to_cstr(&properties.device_name).to_str().unwrap_or("unknown").to_owned();

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

    pub fn query_surface_info(&self, phys_dev: vk::PhysicalDevice, surface: vk::SurfaceKHR) -> VulkanResult<SurfaceInfo> {
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

    pub fn create_logical_device(&self, dev_info: &DeviceInfo) -> VulkanResult<ash::Device> {
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

impl std::ops::Deref for VulkanInstance {
    type Target = ash::Instance;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub phys_dev: vk::PhysicalDevice,
    pub name: String,
    pub dev_type: DeviceType,
    pub graphics_idx: u32,
    pub present_idx: u32,
    pub extensions: HashSet<CString>,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
}

impl DeviceInfo {
    pub fn unique_families(&self) -> Vec<u32> {
        if self.graphics_idx == self.present_idx {
            vec![self.graphics_idx]
        } else {
            vec![self.graphics_idx, self.present_idx]
        }
    }

    pub fn find_memory_type(&self, type_filter: u32, prop_flags: vk::MemoryPropertyFlags) -> VulkanResult<u32> {
        for i in 0..self.mem_props.memory_type_count {
            if type_filter & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(prop_flags) {
                return Ok(i);
            }
        }
        Err(VkError::EngineError("Failed to find suitable memory type"))
    }
}

#[derive(Debug, Clone)]
pub struct SurfaceInfo {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SurfaceInfo {
    pub fn surface_format(&self) -> &vk::SurfaceFormatKHR {
        self.formats
            .iter()
            .find(|&fmt| fmt.format == vk::Format::B8G8R8A8_SRGB && fmt.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .or_else(|| self.formats.first())
            .expect("Empty surface formats")
    }

    pub fn present_mode(&self) -> vk::PresentModeKHR {
        self.present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::IMMEDIATE)
            .unwrap_or(vk::PresentModeKHR::FIFO)
    }

    pub fn calc_extent(&self, width: u32, height: u32) -> vk::Extent2D {
        if self.capabilities.current_extent.width != u32::max_value() {
            self.capabilities.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(self.capabilities.min_image_extent.width, self.capabilities.max_image_extent.width),
                height: height.clamp(self.capabilities.min_image_extent.height, self.capabilities.max_image_extent.height),
            }
        }
    }

    pub fn calc_image_count(&self, count: u32) -> u32 {
        if self.capabilities.max_image_count > 0 {
            count.clamp(self.capabilities.min_image_count, self.capabilities.max_image_count)
        } else {
            count.max(self.capabilities.min_image_count)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum DeviceType {
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct DeviceSelection<'a> {
    name: Option<&'a str>,
    dev_type: Option<DeviceType>,
}

impl DeviceSelection<'_> {
    fn is_empty(&self) -> bool {
        self.name.is_none() && self.dev_type.is_none()
    }

    fn matches(&self, dev_info: &DeviceInfo) -> bool {
        [
            self.name.map(|name| dev_info.name.contains(name)),
            self.dev_type.map(|ty| dev_info.dev_type == ty),
        ]
        .into_iter()
        .flatten()
        .all(identity)
    }
}

impl<'a> From<&'a str> for DeviceSelection<'a> {
    #[inline]
    fn from(name: &'a str) -> Self {
        Self {
            name: Some(name),
            dev_type: None,
        }
    }
}

impl From<DeviceType> for DeviceSelection<'_> {
    #[inline]
    fn from(ty: DeviceType) -> Self {
        Self {
            name: None,
            dev_type: Some(ty),
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
