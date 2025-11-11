use crate::types::Cleanup;
use ash::ext;
use ash::vk::{self, Handle};
use std::ffi::{CStr, CString, c_void};

pub struct DebugUtils {
    debug_dfn: ext::debug_utils::Device,
}

impl DebugUtils {
    pub fn new(instance: &ash::Instance, device: &ash::Device) -> Self {
        let debug_dfn = ext::debug_utils::Device::new(instance, device);
        Self { debug_dfn }
    }

    pub fn set_object_name<O: Handle + Copy>(&self, object: O, name: &str) {
        let name = CString::new(name).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::default().object_handle(object).object_name(&name);
        unsafe {
            self.debug_dfn
                .set_debug_utils_object_name(&name_info)
                .expect("Failed to set DebugUtils object name");
        }
    }

    pub fn set_object_tag<O: Handle + Copy>(&self, object: O, tag_name: u64, tag: &[u8]) {
        let tag_info = vk::DebugUtilsObjectTagInfoEXT::default()
            .object_handle(object)
            .tag_name(tag_name)
            .tag(tag);
        unsafe {
            self.debug_dfn
                .set_debug_utils_object_tag(&tag_info)
                .expect("Failed to set DebugUtils object tag");
        }
    }

    pub fn cmd_begin_label(&self, cmd_buffer: vk::CommandBuffer, name: &str, color: [f32; 4]) {
        let name = CString::new(name).unwrap();
        let label_info = vk::DebugUtilsLabelEXT::default().label_name(&name).color(color);
        unsafe {
            self.debug_dfn.cmd_begin_debug_utils_label(cmd_buffer, &label_info);
        }
    }

    pub fn cmd_insert_label(&self, cmd_buffer: vk::CommandBuffer, name: &str, color: [f32; 4]) {
        let name = CString::new(name).unwrap();
        let label_info = vk::DebugUtilsLabelEXT::default().label_name(&name).color(color);
        unsafe {
            self.debug_dfn.cmd_insert_debug_utils_label(cmd_buffer, &label_info);
        }
    }

    pub fn cmd_end_label(&self, cmd_buffer: vk::CommandBuffer) {
        unsafe {
            self.debug_dfn.cmd_end_debug_utils_label(cmd_buffer);
        }
    }

    pub fn queue_begin_label(&self, queue: vk::Queue, name: &str, color: [f32; 4]) {
        let name = CString::new(name).unwrap();
        let label_info = vk::DebugUtilsLabelEXT::default().label_name(&name).color(color);
        unsafe {
            self.debug_dfn.queue_begin_debug_utils_label(queue, &label_info);
        }
    }

    pub fn queue_insert_label(&self, queue: vk::Queue, name: &str, color: [f32; 4]) {
        let name = CString::new(name).unwrap();
        let label_info = vk::DebugUtilsLabelEXT::default().label_name(&name).color(color);
        unsafe { self.debug_dfn.queue_insert_debug_utils_label(queue, &label_info) }
    }

    pub fn queue_end_label(&self, queue: vk::Queue) {
        unsafe {
            self.debug_dfn.queue_end_debug_utils_label(queue);
        }
    }
}

pub struct DebugUtilsInstance {
    debug_ifn: ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl DebugUtilsInstance {
    pub fn new(entry: &ash::Entry, instance: &ash::Instance) -> Self {
        let debug_ifn = ext::debug_utils::Instance::new(entry, instance);
        let dbg_messenger_ci = Self::create_debug_messenger_ci();
        let messenger = unsafe {
            debug_ifn
                .create_debug_utils_messenger(&dbg_messenger_ci, None)
                .expect("Error creating debug utils callback")
        };
        Self { debug_ifn, messenger }
    }

    pub fn create_debug_messenger_ci() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
        vk::DebugUtilsMessengerCreateInfoEXT::default()
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
    }
}

impl Cleanup<()> for DebugUtilsInstance {
    unsafe fn cleanup(&mut self, _: &()) {
        unsafe {
            self.debug_ifn.destroy_debug_utils_messenger(self.messenger, None);
        }
    }
}

extern "system" fn vulkan_debug_utils_callback(
    msg_severity: vk::DebugUtilsMessageSeverityFlagsEXT, msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    cb_data: *const vk::DebugUtilsMessengerCallbackDataEXT, _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*cb_data).p_message) }.to_string_lossy();
    eprintln!("{msg_severity:?} [{msg_type:?}]: {message}");
    vk::FALSE
}
