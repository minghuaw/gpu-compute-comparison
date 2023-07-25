use std::ops::{Add, AddAssign};

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
    VulkanLibrary,
};

mod kernels;

fn main() {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let instance =
        Instance::new(library, InstanceCreateInfo::default()).expect("failed to create instance");
    let physical_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
        .expect("no devices available");
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .expect("couldn't find a graphical queue family") as u32;
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            // here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("failed to create device");
    let queue = queues.next().unwrap();

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    const M: usize = 4096;
    const N: usize = 4096;
    const K: usize = 4096;

    let matrix_a = generate_row_major_matrix::<i32, M, K>(1);
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
        matrix_a,
    )
    .expect("failed to create buffer");

    let matrix_b = generate_row_major_matrix::<i32, K, N>(1);
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
        matrix_b,
    )
    .expect("failed to create buffer");

    let matrix_c = vec![0i32; N * M];
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
    )
    .expect("failed to create buffer");

    let shader = kernels::matmul::naive::load(device.clone()).expect("failed to create shader module");
    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .expect("failed to create compute pipeline");

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, matrix_a_buf.clone()),
            WriteDescriptorSet::buffer(1, matrix_b_buf.clone()),
            WriteDescriptorSet::buffer(2, matrix_c_buf.clone()),
        ],
    )
    .unwrap();

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

    let x = if M % 8 == 0 { M / 8 } else { M / 8 + 1 } as u32;
    let y = if N % 8 == 0 { N / 8 } else { N / 8 + 1 } as u32;
    let work_group_counts = [x, y, 1];

    command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )
        .dispatch(work_group_counts)
        .unwrap();

    let command_buffer = command_buffer_builder.build().unwrap();

    let start = std::time::Instant::now();
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();
    let elapsed = start.elapsed();
    println!("Elapsed: {:?}", elapsed);

    let content = matrix_c_buf.read().unwrap();
    println!("Element: {}", content[M * N - 1]);
    // println!("Content: {:?}", &content[..]);

    println!("Everything succeeded!");
}

pub(crate) fn generate_matrix<T, const R: usize, const C: usize>(step: T) -> Vec<[T; C]>
where
    T: Copy + Default + Add<T> + AddAssign<T>,
{
    let mut ret = vec![[T::default(); C]; R];
    let mut next = T::default();

    ret.iter_mut()
        .for_each(|row| row.iter_mut().for_each(|elem| {
            *elem = next;
            next += step;
        }));
    ret
}

fn generate_row_major_matrix<T, const R: usize, const C: usize>(step: T) -> Vec<T>
where
    T: Copy + Default + Add<T> + AddAssign<T>,
{
    let mut ret = vec![T::default(); R * C];
    let mut next = T::default();

    ret.iter_mut().for_each(|elem| {
        *elem = next;
        next += step;
    });
    ret
}