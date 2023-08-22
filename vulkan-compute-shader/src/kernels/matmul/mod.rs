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

pub(crate) mod tiling {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/tiling.comp"
    }
}

pub(crate) mod block_tiling_1d {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/block_tiling_1d.comp"
    }
}