pub(crate) mod naive {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./shaders/matmul/naive.comp"
    }
}