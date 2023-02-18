use crate::types::{Cleanup, ObjectType};
use ash::extensions::ext;
use ash::vk::{self, Handle};
use std::ffi::{c_void, CStr, CString};

pub struct DebugUtils {
    debug_fn: ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl DebugUtils {
    pub fn new(entry: &ash::Entry, instance: &ash::Instance) -> Self {
        let utils = ext::DebugUtils::new(entry, instance);
        let dbg_messenger_ci = Self::create_debug_messenger_ci();
        let messenger = unsafe {
            utils
                .create_debug_utils_messenger(&dbg_messenger_ci, None)
                .expect("Error creating debug utils callback")
        };
        Self {
            debug_fn: utils,
            messenger,
        }
    }

    pub fn create_debug_messenger_ci() -> vk::DebugUtilsMessengerCreateInfoEXT {
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

    pub fn set_object_name<O>(&self, device: &ash::Device, object: &O, name: &str)
    where
        O: ObjectType + Handle + Copy,
    {
        let name = CString::new(name).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_handle(object.as_raw())
            .object_type(O::VK_OBJECT_TYPE)
            .object_name(&name);
        unsafe {
            self.debug_fn
                .set_debug_utils_object_name(device.handle(), &name_info)
                .expect("Failed to set DebugUtils object name");
        }
    }

    pub fn set_object_tag<O>(&self, device: &ash::Device, object: &O, tag_name: u64, tag: &[u8])
    where
        O: ObjectType + Handle + Copy,
    {
        let tag_info = vk::DebugUtilsObjectTagInfoEXT::builder()
            .object_handle(object.as_raw())
            .object_type(O::VK_OBJECT_TYPE)
            .tag_name(tag_name)
            .tag(tag);
        unsafe {
            self.debug_fn
                .set_debug_utils_object_tag(device.handle(), &tag_info)
                .expect("Failed to set DebugUtils object tag");
        }
    }

    pub fn cmd_begin_label(&self, cmd_buffer: vk::CommandBuffer, name: &str, color: [f32; 4]) {
        let name = CString::new(name).unwrap();
        let label_info = vk::DebugUtilsLabelEXT::builder().label_name(&name).color(color);
        unsafe {
            self.debug_fn.cmd_begin_debug_utils_label(cmd_buffer, &label_info);
        }
    }

    pub fn cmd_insert_label(&self, cmd_buffer: vk::CommandBuffer, name: &str, color: [f32; 4]) {
        let name = CString::new(name).unwrap();
        let label_info = vk::DebugUtilsLabelEXT::builder().label_name(&name).color(color);
        unsafe {
            self.debug_fn.cmd_insert_debug_utils_label(cmd_buffer, &label_info);
        }
    }

    pub fn cmd_end_label(&self, cmd_buffer: vk::CommandBuffer) {
        unsafe {
            self.debug_fn.cmd_end_debug_utils_label(cmd_buffer);
        }
    }

    pub fn queue_begin_label(&self, queue: vk::Queue, name: &str, color: [f32; 4]) {
        let name = CString::new(name).unwrap();
        let label_info = vk::DebugUtilsLabelEXT::builder().label_name(&name).color(color);
        unsafe {
            self.debug_fn.queue_begin_debug_utils_label(queue, &label_info);
        }
    }

    pub fn queue_insert_label(&self, queue: vk::Queue, name: &str, color: [f32; 4]) {
        let name = CString::new(name).unwrap();
        let label_info = vk::DebugUtilsLabelEXT::builder().label_name(&name).color(color);
        unsafe { self.debug_fn.queue_insert_debug_utils_label(queue, &label_info) }
    }

    pub fn queue_end_label(&self, queue: vk::Queue) {
        unsafe {
            self.debug_fn.queue_end_debug_utils_label(queue);
        }
    }
}

impl Cleanup<()> for DebugUtils {
    unsafe fn cleanup(&mut self, _: &()) {
        self.debug_fn.destroy_debug_utils_messenger(self.messenger, None);
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
