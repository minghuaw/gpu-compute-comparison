use std::sync::Arc;

use vulkano::{
    device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags},
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary,
};

pub(crate) const M: usize = 4096;
pub(crate) const N: usize = 4096;
pub(crate) const K: usize = 4096;

pub(crate) type BoxError = Box<dyn std::error::Error>;

pub(crate) fn get_device_and_queue() -> Result<(Arc<Device>, Arc<Queue>), BoxError> {
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

    Ok((device, queue))
}
