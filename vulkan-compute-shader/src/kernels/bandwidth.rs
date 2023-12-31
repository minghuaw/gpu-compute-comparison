use std::{sync::Arc, time::Duration};

use ndarray::Array2;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferError, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    shader::ShaderModule,
    sync::GpuFuture,
};

use crate::common::{BoxError, M, N};

pub(crate) mod block_to_global_1d {
    //! This kernel simply copies from the 1d shared memory block to the global memory
    use super::*;

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/bandwidth/block_to_global_1d.comp"
    }

    const BM: usize = 32;

    pub(crate) fn run(device: Arc<Device>, queue: Arc<Queue>) -> Result<Duration, BoxError> {
        let shader = self::load(device.clone())?;
        super::run(device, queue, shader, [(M / BM) as u32, N as u32, 1])
    }
}

pub(crate) mod block_to_global_2d {
    //! This kernel simply copies from the 2d shared memory block to the global memory
    use super::*;

    const BM: usize = 32;
    const BN: usize = 32;

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/bandwidth/block_to_global_2d.comp"
    }

    pub(crate) fn run(device: Arc<Device>, queue: Arc<Queue>) -> Result<Duration, BoxError> {
        let shader = self::load(device.clone())?;
        super::run(device, queue, shader, [(M / BM) as u32, (N / BN) as u32, 1])
    }
}

pub(crate) mod tile_to_global_1d {
    //! This kernel simply copies from the 1d tile to the global memory. The tile
    //! is a small 1d array that is stored in the register file.

    use super::*;

    const BM: usize = 64;
    const BN: usize = 64;

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/bandwidth/tile_to_global_1d.comp"
    }

    pub(crate) fn run(device: Arc<Device>, queue: Arc<Queue>) -> Result<Duration, BoxError> {
        let shader = self::load(device.clone())?;
        super::run(device, queue, shader, [(M / BM) as u32, (N / BN) as u32, 1])
    }
}

pub(crate) mod tile_to_global_2d {
    //! This kernel simply copies from the 2d tile to the global memory. The tile
    //! is a small 2d array that is stored in the register file.
    
    use super::*;

    const BM: usize = 64;
    const BN: usize = 64;

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/bandwidth/tile_to_global_2d.comp"
    }

    pub(crate) fn run(device: Arc<Device>, queue: Arc<Queue>) -> Result<Duration, BoxError> {
        let shader = self::load(device.clone())?;
        super::run(device, queue, shader, [(M / BM) as u32, (N / BN) as u32, 1])
    }
}

pub(crate) mod tile_to_block_to_global_1d {
    //! This kernel fisrt copies from tile to shared memory block and then copies
    //! from the shared memory block to the global memory.
    
    use super::*;

    const BM: usize = 64;
    const TM: usize = 8;

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/bandwidth/tile_to_block_to_global_1d.comp"
    }

    pub(crate) fn run(device: Arc<Device>, queue: Arc<Queue>) -> Result<Duration, BoxError> {
        let shader = self::load(device.clone())?;
        super::run(device, queue, shader, [(M / BM) as u32, N as u32, 1])
    }
}

pub(crate) mod tile_to_block_to_global_2d {
    //! This kernel fisrt copies from tile to shared memory block and then copies
    //! from the shared memory block to the global memory.
}

pub(crate) mod vec4_to_global {
    use super::*;

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/bandwidth/vec4_to_global.comp"
    }

    pub(crate) fn run(device: Arc<Device>, queue: Arc<Queue>) -> Result<Duration, BoxError> {
        let shader = self::load(device.clone())?;
        super::run(device, queue, shader, [M as u32, N as u32 / 4, 1])
    }
}

fn allocate_output_buffer(
    allocator: &StandardMemoryAllocator,
) -> Result<Subbuffer<[f32]>, BufferError> {
    let zeros = Array2::zeros((M, N));
    Buffer::from_iter(
        allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Download,
            ..Default::default()
        },
        zeros,
    )
}

fn run(
    device: Arc<Device>,
    queue: Arc<Queue>,
    shader: Arc<ShaderModule>,
    group_counts: [u32; 3],
) -> Result<Duration, BoxError> {
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
    let output_buffer = allocate_output_buffer(&memory_allocator)?;

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader
            .entry_point("main")
            .ok_or("failed to get entry point")?,
        &(),
        None,
        |_| {},
    )?;
    let pipeline_layout = compute_pipeline.layout();

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .ok_or("failed to get descriptor set layout")?;
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, output_buffer.clone())],
    )?;

    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());
    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )
        .dispatch(group_counts)?;
    let command_buffer = command_buffer_builder.build()?;

    let start = std::time::Instant::now();
    let future = vulkano::sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)?
        .then_signal_fence_and_flush()?;
    future.wait(None)?;
    let elapsed = start.elapsed();

    let guard = output_buffer.read()?;
    // println!("[{}]", guard[100*4096]);
    assert!(guard.iter().all(|&x| x != 0.0)); // All elements should be non-zero
    // for (i, &x) in guard.iter().enumerate() {
    //     if x == 0.0 {
    //         println!("zero at index {}", i);
    //         break;
    //     }
    // }

    Ok(elapsed)
}
