#[unsafe(no_mangle)]
pub static DIMENSIONS: usize = ese::DIMENSIONS;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn encode(input: *const std::ffi::c_char, out: *mut f32) {
    unsafe {
        let cstr = std::ffi::CStr::from_ptr(input);
        let s = cstr.to_str().unwrap_or("");
        let embedding = ese::encode_single(s);
        std::ptr::copy_nonoverlapping(embedding.as_ptr(), out, DIMENSIONS);
    }
}
