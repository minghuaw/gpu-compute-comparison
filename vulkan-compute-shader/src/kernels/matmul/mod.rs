pub(crate) mod naive {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/naive.comp"
    }
}

pub (crate) mod cache_blocking {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/cache_blocking.comp"
    }
}