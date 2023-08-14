use std::path::PathBuf;

use metal::*;
use ndarray::{ArrayBase, OwnedRepr};
use ndarray_rand::RandomExt;
use objc::rc::autoreleasepool;
use rand::distributions::Uniform;

fn main() {
    autoreleasepool(|| {
        let device = Device::system_default().unwrap();
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();

        let compute_pass_descriptor = ComputePassDescriptor::new();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders/matmul.metallib");
        let library = device.new_library_with_file(library_path).unwrap();
        // let kernel = library.get_function("naive", None).unwrap();
        let kernel = library.get_function("cache_blocking", None).unwrap();

        let pipepine_state_descriptor = ComputePipelineDescriptor::new();
        pipepine_state_descriptor.set_compute_function(Some(&kernel));

        let pipeline_state = device
            .new_compute_pipeline_state_with_function(
                pipepine_state_descriptor.compute_function().unwrap(),
            )
            .unwrap();
        encoder.set_compute_pipeline_state(&pipeline_state);

        const M: usize = 4096;
        const N: usize = 4096;
        const K: usize = 4096;
        const BM: usize = 32;
        const BN: usize = 32;
        const BK: usize = 32;

        let matrix_a: ArrayBase<OwnedRepr<f32>, _> =
            ArrayBase::random((M, K), Uniform::new(-1.0, 1.0));
        let matrix_a_buf = device.new_buffer_with_data(
            unsafe { std::mem::transmute(matrix_a.as_ptr()) },
            (matrix_a.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        let matrix_b: ArrayBase<OwnedRepr<f32>, _> =
            ArrayBase::random((K, N), Uniform::new(-1.0, 1.0));
        let matrix_b_buf = device.new_buffer_with_data(
            unsafe { std::mem::transmute(matrix_b.as_ptr()) },
            (matrix_b.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        let matrix_c: ArrayBase<OwnedRepr<f32>, _> = ArrayBase::zeros((M, N));
        let matrix_c_buf = device.new_buffer_with_data(
            unsafe { std::mem::transmute(matrix_c.as_ptr()) },
            (matrix_c.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        encoder.set_buffer(0, Some(&matrix_a_buf), 0);
        encoder.set_buffer(1, Some(&matrix_b_buf), 0);
        encoder.set_buffer(2, Some(&matrix_c_buf), 0);

        // let num_threads = pipeline_state.thread_execution_width();

        let width = if M % BM == 0 { M / BM } else { M / BM + 1 } as u64;
        let height = if N % BN == 0 { N / BN } else { N / BN + 1 } as u64;
        let thread_groups_count = MTLSize {
            width,
            height,
            depth: 1,
        };
        let threads_per_threadgroup = MTLSize {
            width: BM,
            height: BN,
            depth: 1,
        };
        encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
        encoder.end_encoding();
        let start = std::time::Instant::now();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let elapsed = start.elapsed();
        println!("Elapsed: {:?}", elapsed);

        // // Visualize the result
        // unsafe {
        //     let result = matrix_c_buf.contents() as *mut f32;
        //     println!("Result: {}", *result.offset(0));
        //     println!("Result: {}", *result.offset(1));
        //     println!("Result: {}", *result.offset(matrix_c.len() as isize - 1));
        // }
    })
}
