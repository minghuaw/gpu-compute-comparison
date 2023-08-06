use std::path::PathBuf;

use metal::*;
use objc::rc::autoreleasepool;

fn main() {
    autoreleasepool(|| {
        let device = Device::system_default().unwrap();
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();

        let compute_pass_descriptor = ComputePassDescriptor::new();
        let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders/matmul.metallib");
        let library = device.new_library_with_file(library_path).unwrap();
        let kernel = library.get_function("add", None).unwrap();

        let pipepine_state_descriptor = ComputePipelineDescriptor::new();
        pipepine_state_descriptor.set_compute_function(Some(&kernel));

        let pipeline_state = device
            .new_compute_pipeline_state_with_function(
                pipepine_state_descriptor.compute_function().unwrap(),
            )
            .unwrap();
        encoder.set_compute_pipeline_state(&pipeline_state);

        // Create input and output buffers
        let input = [1.0f32, 2.0];
        let input_buffer = device.new_buffer_with_data(
            unsafe {std::mem::transmute(input.as_ptr())}, 
            (input.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let output_buffer = device.new_buffer(
            std::mem::size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        // let num_threads = pipeline_state.thread_execution_width();

        let thread_groups_count = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe {
            let result = output_buffer.contents() as *mut f32;
            println!("Result: {}", *result);
        }
    })
}
