use std::{borrow::Cow, ffi::CStr};

use ash::vk;

pub unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    cb_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let cb_data = *cb_data;

    let name = if cb_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(cb_data.p_message_id_name).to_string_lossy()
    };

    let message = if cb_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(cb_data.p_message).to_string_lossy()
    };

    let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        log::Level::Info
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::Level::Warn
    } else {
        log::Level::Error
    };

    log::log!(level, "{:?} [{}]: {}", ty, name, message);
    if std::env::var("REX_VK_BACKTRACE").is_ok() {
        if level == log::Level::Error {
            log::error!(
                "Backtrace: {}",
                std::backtrace::Backtrace::capture().to_string()
            );
        }
    }

    vk::FALSE
}
