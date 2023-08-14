use ndarray::{ArrayBase, OwnedRepr};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use wgpu::{Backends, util::{DeviceExt, BufferInitDescriptor}, BufferUsages, ComputePipelineDescriptor, Maintain};

mod kernels;

async fn run() {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .enumerate_adapters(Backends::all())
        .filter(|adapter| adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .next()
        .unwrap_or_else(|| {
            instance
                .enumerate_adapters(Backends::all())
                .next()
                .expect("failed to find an adapter")
        });
    let limits = wgpu::Limits {
        max_compute_invocations_per_workgroup: 1024,
        ..Default::default()
    };
    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::empty(),
            limits,
        },
        None,
    ).await
    .expect("failed to create device");

    const M: usize = 4096;
    const N: usize = 4096;
    const K: usize = 4096;
    const BM: usize = 32;
    const BN: usize = 32;

    let matrix_a: ArrayBase<OwnedRepr<f32>, _> = ArrayBase::random((M, K), Uniform::new(-1.0, 1.0));
    let matrix_a_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(matrix_a.as_slice().expect("failed to get slice")),
        usage: BufferUsages::STORAGE,
    });

    let matrix_b: ArrayBase<OwnedRepr<f32>, _> = ArrayBase::random((K, N), Uniform::new(-1.0, 1.0));
    let matrix_b_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(matrix_b.as_slice().expect("failed to get slice")),
        usage: BufferUsages::STORAGE,
    });

    let matrix_c: ArrayBase<OwnedRepr<f32>, _> = ArrayBase::zeros((M, N));
    let matrix_c_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(matrix_c.as_slice().expect("failed to get slice")),
        usage: BufferUsages::STORAGE,
    });

    // let shader = kernels::matmul::naive::load(&device);
    let shader = kernels::matmul::cache_blocking::load(&device);
    let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: "main",
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix_a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: matrix_b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: matrix_c_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: None,
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        let x = if M % BM == 0 { M / BM } else { M / BM + 1 } as u32;
        let y = if N % BN == 0 { N / BN } else { N / BN + 1 } as u32;
        cpass.dispatch_workgroups(x, y, 1);
    }

    let start = std::time::Instant::now();
    queue.submit(Some(encoder.finish()));
    device.poll(Maintain::Wait);
    let elapsed = start.elapsed();
    println!("Elapsed: {:?}", elapsed);
}

fn main() {
    pollster::block_on(run());
}
