use std::{sync::Arc, time::Duration};

use ndarray::{linalg::general_mat_mul, Array2, ArrayBase};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage,
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

use crate::common::{BoxError, K, M, N};

pub(crate) mod naive {
    use super::*;

    const BM: usize = 32;
    const BN: usize = 32;

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/naive.comp"
    }

    pub(crate) fn run(device: Arc<Device>, queue: Arc<Queue>) -> Result<Duration, BoxError> {
        let shader = self::load(device.clone())?;
        super::run(device, queue, shader, [(M / BM) as u32, (N / BN) as u32, 1])
    }
}

pub(crate) mod cache_blocking {
    use super::*;

    const BM: usize = 32;
    const BN: usize = 32;

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/cache_blocking.comp"
    }

    pub(crate) fn run(device: Arc<Device>, queue: Arc<Queue>) -> Result<Duration, BoxError> {
        let shader = self::load(device.clone())?;
        super::run(device, queue, shader, [(M / BM) as u32, (N / BN) as u32, 1])
    }
}

pub(crate) mod tiling {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/tiling.comp"
    }
}

pub(crate) mod block_tiling_1d {
    use super::*;

    const BM: usize = 64;
    const BN: usize = 64;
    const TM: usize = 8;

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/block_tiling_1d.comp"
    }

    pub(crate) fn run(device: Arc<Device>, queue: Arc<Queue>) -> Result<Duration, BoxError> {
        let shader = self::load(device.clone())?;
        super::run(device, queue, shader, [(M / BM) as u32, (N / BN) as u32, 1])
    }
}

pub(crate) mod block_tiling_2d {
    use super::*;

    const BM: usize = 64;
    const BN: usize = 64;

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/block_tiling_2d.comp"
    }

    pub(crate) fn run(device: Arc<Device>, queue: Arc<Queue>) -> Result<Duration, BoxError> {
        let shader = self::load(device.clone())?;
        super::run(device, queue, shader, [(M / BM) as u32, (N / BN) as u32, 1])
    }
}

pub(crate) mod vectorize_block_tiling_2d {
    use super::*;

    const BM: usize = 128;
    const BN: usize = 128;

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/vectorize_block_tiling_2d.comp"
    }

    pub(crate) fn run(device: Arc<Device>, queue: Arc<Queue>) -> Result<Duration, BoxError> {
        let shader = self::load(device.clone())?;
        super::run(device, queue, shader, [(M / BM) as u32, (N / BN) as u32, 1])
    }
}

pub(crate) mod padding {
    use super::*;

    const BM: usize = 128;
    const BN: usize = 128;

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/padding.comp"
    }

    pub(crate) fn run(device: Arc<Device>, queue: Arc<Queue>) -> Result<Duration, BoxError> {
        let shader = self::load(device.clone())?;
        super::run(device, queue, shader, [(M / BM) as u32, (N / BN) as u32, 1])
    }
}

pub(crate) mod write_tile_1d {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/write_tile_1d.comp"
    }
}

pub(crate) mod write_tile_2d {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/write_tile_2d.comp"
    }
}

fn run(
    device: Arc<Device>,
    queue: Arc<Queue>,
    shader: Arc<ShaderModule>,
    group_counts: [u32; 3],
) -> Result<Duration, BoxError> {
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    let matrix_a: Array2<f32> = ArrayBase::random((M, K), Uniform::new(-1.0, 1.0));
    let matrix_b: Array2<f32> = ArrayBase::random((K, N), Uniform::new(-1.0, 1.0));
    let matrix_c: Array2<f32> = ArrayBase::zeros((M, N));

    let matrix_a_buf = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        matrix_a.clone(),
    )?;
    let matrix_b_buf = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        matrix_b.clone(),
    )?;
    let matrix_c_buf = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Download,
            ..Default::default()
        },
        matrix_c,
    )?;

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )?;

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .ok_or("failed to get descriptor set layout")?;
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, matrix_a_buf.clone()),
            WriteDescriptorSet::buffer(1, matrix_b_buf.clone()),
            WriteDescriptorSet::buffer(2, matrix_c_buf.clone()),
        ],
    )?;

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )
        .dispatch(group_counts)?;

    let command_buffer = command_buffer_builder.build().unwrap();

    let start = std::time::Instant::now();
    let future = vulkano::sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()?;
    future.wait(None)?;
    let elapsed = start.elapsed();

    let mut expected: Array2<f32> = ArrayBase::zeros((M, N));
    general_mat_mul(1.0, &matrix_a, &matrix_b, 1.0, &mut expected);
    assert!(is_equal::<M, N>(matrix_c_buf, expected));

    Ok(elapsed)
}

fn is_equal<const M: usize, const N: usize>(
    value: Subbuffer<[f32]>,
    expected: Array2<f32>,
) -> bool {
    let guard = value.read().unwrap();
    let slice = expected.as_slice().unwrap();
    let total = M * N;
    for i in 0..total {
        if f32::abs(guard[i] - slice[i]) > 1e-2 {
            println!(
                "at index {}: value {} != expected {}",
                i, guard[i], slice[i]
            );
            return false;
        }
    }
    true
}
