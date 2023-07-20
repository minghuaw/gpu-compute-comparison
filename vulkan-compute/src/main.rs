//! This is copied from https://gist.github.com/itzmeanjan/84613bc7595372c5e6b6c22481d42f9a

extern crate rand;
extern crate vulkano;
extern crate vulkano_shaders;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

use std::time::Instant;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::instance::PhysicalDevice;
use vulkano::instance::{Instance, InstanceExtensions};
use vulkano::pipeline::{ComputePipeline, ComputePipelineAbstract};
use vulkano::sync::GpuFuture;
use vulkano::Version;

const N: u32 = 1 << 20;

fn main() {
    let instance = Instance::new(None, Version::V1_2, &InstanceExtensions::none(), None)
        .expect("failed to create instance !");
    let physical_device = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("failed to enumerate physical devices");

    println!(
        "Device: {}\nVulkan API: {}",
        physical_device.properties().device_name.as_ref().unwrap(),
        physical_device.api_version()
    );

    for i in physical_device.queue_families() {
        println!(
            "Queue Count: {}\tCompute: {}\tGraphics: {}",
            i.queues_count(),
            i.supports_compute(),
            i.supports_graphics()
        );
    }

    let queue_family = physical_device
        .queue_families()
        .find(|&v| v.supports_compute())
        .expect("failed to find compute supported queue family");

    let mut ext = DeviceExtensions::none();
    ext.khr_storage_buffer_storage_class = true;
    let (logical_device, mut queues) = Device::new(
        physical_device,
        &Features::none(),
        &ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .expect("failed to create logical logical_device");
    let queue = queues.next().expect("failed to find associated queue");

    let matrix_a = generate_square_matrix(Some(13));
    let matrix_b = generate_square_matrix(Some(17));
    let matrix_c = generate_square_matrix(None);
    
    // Matrix A --- stored in GPU accessible memory, CPU can't access it
    let (matrix_a_buf, _) = ImmutableBuffer::from_iter(matrix_a, BufferUsage::all(), queue.clone())
        .expect("failed to create uniform buffer");
    // Matrix B --- stored in GPU accessible memory, CPU can't access it
    let (matrix_b_buf, _) = ImmutableBuffer::from_iter(matrix_b, BufferUsage::all(), queue.clone())
        .expect("failed to create uniform buffer");
    // Matrix C --- resulting matrix can be accessed by both CPU, GPU
    let matrix_c_buf =
        CpuAccessibleBuffer::from_iter(logical_device.clone(), BufferUsage::all(), false, matrix_c)
            .expect("failed to create storage buffer");

    // loading compute shader, including shader compilation
    // abstracted with macro!
    let shader = cs::Shader::load(logical_device.clone()).unwrap();
    // preparing compute pipeline
    let compute_pipeline = Arc::new(
        ComputePipeline::new(
            logical_device.clone(),
            &shader.main_entry_point(),
            &(),
            None,
        )
        .unwrap(),
    );

    // adding descriptors as per layout, into compute pipeline
    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(matrix_a_buf.clone())
            .unwrap()
            .add_buffer(matrix_b_buf.clone())
            .unwrap()
            .add_buffer(matrix_c_buf.clone())
            .unwrap()
            .build()
            .unwrap(),
    );
    // create command buffer & start recording commands in it
    let mut builder = AutoCommandBufferBuilder::primary(
        logical_device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    // only single command recorded in command buffer
    builder
        .dispatch(
            [1024 / 8, 1024 / 4, 1],
            compute_pipeline.clone(),
            set.clone(),
            (),
            std::iter::empty(),
        )
        .unwrap();
    // ending command recording
    let command_buffer = builder.build().unwrap();
    
    // Computing Matrix Multiplication on GPU
    let start = Instant::now();
    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
    let gpu_tm = start.elapsed();
    println!("GPU matrix multiply: {:?}", gpu_tm);
    
    let r_matrix_a = generate_square_matrix(Some(13)).collect::<Vec<i32>>();
    let r_matrix_b = generate_square_matrix(Some(17)).collect::<Vec<i32>>();
    // reading GPU-computed matrix multiplication result
    let gpu_result = matrix_c_buf.read().unwrap();
    
    // Computing Matrix Multiplication on CPU, and asserting !
    let start = Instant::now();
    for i in 0..1024 {
        for j in 0..1024 {
            let mut sum = 0i32;
            for k in 0..1024 {
                sum += r_matrix_a[i * 1024 + k] * r_matrix_b[k * 1024 + j];
            }
            assert_eq!(sum, gpu_result[i * 1024 + j]);
        }
    }
    println!(
        "CPU matrix multiply: {:?}\nSpeed Up: {}",
        start.elapsed(),
        start.elapsed().as_nanos() / gpu_tm.as_nanos()
    );
}

// reproducible random matrix generator, as single dimensional iterator
fn generate_square_matrix(seed: Option<u64>) -> Box<dyn std::iter::ExactSizeIterator<Item = i32>> {
    match seed {
        Some(seed) => {
            let mut rng = StdRng::seed_from_u64(seed);
            Box::new((0..N).map(move |_| rng.gen_range(0..10)))
        }
        None => Box::new((0..N).map(|_| 0)),
    }
}

mod cs {
    // does shader compilation
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./matrix_multiply.glsl",
        vulkan_version: "1.2",
    }
}
