#![allow(dead_code)]

pub(crate) mod naive {
    use std::borrow::Cow;

    use wgpu::{Device, ShaderModule};

    pub(crate) fn load(device: &Device) -> ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_naive"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../../shaders/matmul/naive.wgsl"
            ))),
        })
    }
}

pub(crate) mod cache_blocking {
    use std::borrow::Cow;

    use wgpu::{Device, ShaderModule};

    pub(crate) fn load(device: &Device) -> ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_cache_blocking"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../../shaders/matmul/cache_blocking.wgsl"
            ))),
        })
    }
}