use crate::debug::DebugUtilsInstance;
use crate::types::*;
use ash::vk;
use ash::{ext, khr};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::collections::BTreeSet;
use std::convert::identity;
use std::ffi::{c_char, CStr, CString};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";
const DEVICE_EXTENSIONS: [(&CStr, bool); 4] = [
    (khr::swapchain::NAME, true),
    (khr::portability_subset::NAME, false),
    (khr::dynamic_rendering::NAME, true),
    (khr::push_descriptor::NAME, true),
];
const VULKAN_VERSION: u32 = vk::API_VERSION_1_2;

pub(crate) struct VulkanInstance {
    entry: ash::Entry,
    instance: ash::Instance,
    pub surface_utils: khr::surface::Instance,
    #[cfg(debug_assertions)]
    debug_utils: DebugUtilsInstance,
}

impl VulkanInstance {
    pub fn new<W: HasDisplayHandle>(window: &W, app_name: &str) -> VulkanResult<Self> {
        let entry = unsafe { ash::Entry::load()? };
        if VALIDATION_ENABLED {
            Self::check_validation_support(&entry)?;
        }
        let app_name = CString::new(app_name).unwrap();
        let engine_name = CString::new(env!("CARGO_PKG_NAME")).unwrap();
        let instance = Self::create_instance(&entry, window, &app_name, &engine_name)?;
        let surface_utils = khr::surface::Instance::new(&entry, &instance);

        Ok(Self {
            #[cfg(debug_assertions)]
            debug_utils: DebugUtilsInstance::new(&entry, &instance),
            entry,
            instance,
            surface_utils,
        })
    }

    fn check_validation_support(entry: &ash::Entry) -> VulkanResult<()> {
        let supported_layers = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .describe_err("Failed to enumerate instance layer properties")?
        };
        //eprintln!("Supported instance layers: {supported_layers:#?}");
        supported_layers
            .iter()
            .map(|layer| layer.layer_name_as_c_str().unwrap_or_default())
            .any(|name| name == VALIDATION_LAYER)
            .then_some(())
            .ok_or(VkError::EngineError("Validation layers requested but not available"))
    }

    fn check_portability_support(entry: &ash::Entry) -> VulkanResult<Option<*const c_char>> {
        let ext_list = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .describe_err("Failed to enumerate instance extension properties")?
        };
        //eprintln!("Supported instance extensions: {ext_list:#?}");
        let ext_name = khr::portability_enumeration::NAME;
        let supported = ext_list
            .iter()
            .map(|ext| ext.extension_name_as_c_str().unwrap_or_default())
            .any(|name| name == ext_name)
            .then_some(ext_name.as_ptr());
        Ok(supported)
    }

    fn create_instance<W: HasDisplayHandle>(
        entry: &ash::Entry, window: &W, app_name: &CStr, engine_name: &CStr,
    ) -> VulkanResult<ash::Instance> {
        let version = unsafe { entry.try_enumerate_instance_version()?.unwrap_or(vk::API_VERSION_1_0) };
        if version < VULKAN_VERSION {
            eprintln!(
                "Instance supported Vulkan version {} is lower than required version {}",
                VkVersion(version),
                VkVersion(VULKAN_VERSION)
            );
            return Err(vk::Result::ERROR_INCOMPATIBLE_DRIVER.into());
        }

        let mut extension_names = ash_window::enumerate_required_extensions(window.display_handle()?.into())
            .describe_err("Unsupported display platform")?
            .to_vec();
        let portability = Self::check_portability_support(entry)?;
        extension_names.extend(portability);
        let flags = portability
            .map(|_| vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
            .unwrap_or_default();
        if VALIDATION_ENABLED {
            extension_names.push(ext::debug_utils::NAME.as_ptr());
        }
        for &name in &extension_names {
            eprintln!("Using instance extension: {:?}", unsafe { CStr::from_ptr(name) });
        }

        let layer_names = [VALIDATION_LAYER.as_ptr()];

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(VULKAN_VERSION);

        let mut dbg_messenger_ci = DebugUtilsInstance::create_debug_messenger_ci();
        let mut instance_ci = vk::InstanceCreateInfo::default()
            .flags(flags)
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);
        if VALIDATION_ENABLED {
            instance_ci = instance_ci.enabled_layer_names(&layer_names).push_next(&mut dbg_messenger_ci);
            eprintln!("Using instance layer {VALIDATION_LAYER:?}");
        }

        unsafe { entry.create_instance(&instance_ci, None).describe_err("Failed to create instance") }
    }

    pub fn create_surface<W>(&self, window: &W) -> VulkanResult<vk::SurfaceKHR>
    where
        W: HasDisplayHandle + HasWindowHandle,
    {
        unsafe {
            ash_window::create_surface(
                &self.entry,
                &self.instance,
                window.display_handle()?.into(),
                window.window_handle()?.into(),
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
        let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
        let mut props = vk::PhysicalDeviceProperties2 {
            p_next: &mut driver_props as *mut vk::PhysicalDeviceDriverProperties as *mut _, // FIXME: workaround for .push_next() holding mut borrow to driver_props
            ..Default::default()
        };
        unsafe { self.instance.get_physical_device_properties2(phys_dev, &mut props) };
        let dev_type = props.properties.device_type.into();
        let name = props
            .properties
            .device_name_as_c_str()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let driver = [
            driver_props.driver_name_as_c_str().unwrap_or_default().to_string_lossy(),
            driver_props.driver_info_as_c_str().unwrap_or_default().to_string_lossy(),
        ]
        .join(" ");
        let limits = &props.properties.limits;
        let msaa_support = limits.framebuffer_color_sample_counts & limits.framebuffer_depth_sample_counts;

        // features
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut portability = vk::PhysicalDevicePortabilitySubsetFeaturesKHR::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut features12)
            .push_next(&mut portability);
        unsafe { self.instance.get_physical_device_features2(phys_dev, &mut features) };
        if features.features.sampler_anisotropy == vk::FALSE {
            eprintln!("Device '{name}' doesn't support sampler anisotropy");
            return Err(VkError::UnsuitableDevice);
        }
        if features.features.shader_storage_image_write_without_format == vk::FALSE {
            eprintln!("Device '{name}' doesn't support shader storage image write without format");
            return Err(VkError::UnsuitableDevice);
        }
        if features12.timeline_semaphore == vk::FALSE {
            eprintln!("Device '{name}' doesn't support timeline semaphore");
            return Err(VkError::UnsuitableDevice);
        }
        if features12.host_query_reset == vk::FALSE {
            eprintln!("Device '{name}' doesn't support host query reset");
            return Err(VkError::UnsuitableDevice);
        }

        // queue families
        let queue_families = unsafe { self.instance.get_physical_device_queue_family_properties(phys_dev) };
        let mut graphics_idx = None;
        let mut present_idx = None;
        for (que_family, idx) in queue_families.iter().zip(0..) {
            if que_family.queue_count == 0 {
                continue;
            }
            if que_family.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE) {
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
            eprintln!("Device '{name}' has incomplete queues (graphics: {graphics_idx:?}, present: {present_idx:?})",);
            return Err(VkError::UnsuitableDevice);
        }
        let mut unique_families = vec![graphics_idx.unwrap(), present_idx.unwrap()];
        unique_families.sort_unstable();
        unique_families.dedup();

        // supported extensions
        let ext_list = unsafe {
            self.instance
                .enumerate_device_extension_properties(phys_dev)
                .describe_err("Failed to enumerate device extensions")?
        };
        let extensions: BTreeSet<_> = ext_list
            .iter()
            .map(|ext| ext.extension_name_as_c_str().unwrap_or_default().to_owned())
            .collect();
        //eprintln!("Supported device extensions: {extensions:#?}");
        let missing_ext: Vec<_> = DEVICE_EXTENSIONS
            .into_iter()
            .filter_map(|(name, required)| required.then_some(name))
            .filter(|&ext_name| !extensions.contains(ext_name))
            .collect();
        if !missing_ext.is_empty() {
            eprintln!("Device '{name}' has missing required extensions: {missing_ext:?}");
            return Err(VkError::UnsuitableDevice);
        }

        if extensions.contains(khr::portability_subset::NAME) && portability.image_view_format_swizzle == vk::FALSE {
            eprintln!("Device '{name}' doesn't support image view format swizzle (portability)");
            return Err(VkError::UnsuitableDevice);
        }

        Ok(DeviceInfo {
            phys_dev,
            dev_type,
            name,
            driver,
            graphics_idx: graphics_idx.unwrap(),
            present_idx: present_idx.unwrap(),
            unique_families,
            extensions,
            msaa_support,
            max_aniso: limits.max_sampler_anisotropy,
            timestamp_period: limits.timestamp_period,
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
            .unique_families
            .iter()
            .map(|&idx| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(idx)
                    .queue_priorities(&queue_prio)
            })
            .collect();

        let features = vk::PhysicalDeviceFeatures::default()
            .sampler_anisotropy(true)
            .shader_storage_image_write_without_format(true);

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .host_query_reset(true);

        let extensions: Vec<_> = DEVICE_EXTENSIONS
            .into_iter()
            .filter_map(|(name, required)| {
                required
                    .then_some(name)
                    .or_else(|| dev_info.extensions.contains(name).then_some(name))
            })
            .inspect(|&name| eprintln!("Using device extension {name:?}"))
            .map(CStr::as_ptr)
            .collect();

        let mut dyn_render_enable = vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut portability_feats = vk::PhysicalDevicePortabilitySubsetFeaturesKHR::default().image_view_format_swizzle(true);

        let mut device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues_ci)
            .enabled_features(&features)
            .enabled_extension_names(&extensions)
            .push_next(&mut features12)
            .push_next(&mut dyn_render_enable);

        if dev_info.extensions.contains(khr::portability_subset::NAME) {
            device_ci = device_ci.push_next(&mut portability_feats);
        }

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
            #[cfg(debug_assertions)]
            {
                self.debug_utils.cleanup(&());
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

impl std::fmt::Debug for VulkanInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanInstance").finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub phys_dev: vk::PhysicalDevice,
    pub name: String,
    pub driver: String,
    pub dev_type: DeviceType,
    pub graphics_idx: u32,
    pub present_idx: u32,
    pub unique_families: Vec<u32>,
    pub extensions: BTreeSet<CString>,
    pub msaa_support: vk::SampleCountFlags,
    pub max_aniso: f32,
    pub timestamp_period: f32,
}

impl DeviceInfo {
    pub fn get_max_samples(&self) -> vk::SampleCountFlags {
        let msb = self.msaa_support.as_raw().ilog2();
        vk::SampleCountFlags::from_raw(1 << msb)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SurfaceInfo {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SurfaceInfo {
    pub fn find_surface_format(&self) -> &vk::SurfaceFormatKHR {
        self.formats
            .iter()
            .find(|&fmt| fmt.format == vk::Format::B8G8R8A8_SRGB && fmt.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .or_else(|| self.formats.first())
            .expect("Empty surface formats")
    }

    pub fn find_present_mode(&self, vsync: bool) -> vk::PresentModeKHR {
        let wanted = if vsync {
            vk::PresentModeKHR::FIFO_RELAXED
        } else {
            vk::PresentModeKHR::IMMEDIATE
        };
        self.present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == wanted)
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

#[derive(Debug, Clone, Copy)]
struct VkVersion(u32);

impl std::fmt::Display for VkVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}.{}",
            vk::api_version_major(self.0),
            vk::api_version_minor(self.0),
            vk::api_version_patch(self.0)
        )
    }
}
