use kernels::bandwidth;

use vulkano::{
    device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags},
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary,
};

use crate::kernels::matmul;

extern crate openblas_src;

mod common;
mod kernels;

fn main() {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let create_info = InstanceCreateInfo {
        enumerate_portability: true, // required for MoltenVK on macOS
        ..Default::default()
    };
    let instance = Instance::new(library, create_info).expect("failed to create instance");

    // Prioritize discrete GPUs
    let physical_device = instance
        .enumerate_physical_devices()
        .expect("failed to enumerate devices")
        .filter(|device| {
            matches!(
                device.properties().device_type,
                vulkano::device::physical::PhysicalDeviceType::DiscreteGpu
            )
        })
        .next()
        .unwrap_or_else(|| {
            instance
                .enumerate_physical_devices()
                .expect("failed to enumerate devices")
                .next()
                .expect("no devices available")
        });
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

    let elapsed = bandwidth::block_1d_to_global::run(device.clone(), queue.clone()).unwrap();
    println!("bandwidth::block_1d_to_global elapsed: {:?}", elapsed);

    let elapsed = matmul::naive::run(device.clone(), queue.clone()).unwrap();
    println!("matmul::naive elapsed: {:?}", elapsed);
}
