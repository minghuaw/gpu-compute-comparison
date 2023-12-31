extern crate openblas_src;

mod common;
mod kernels;

fn main() {
    let (device, queue) = common::get_device_and_queue().unwrap();

    // let elapsed = kernels::bandwidth::block_to_global_1d::run(device.clone(), queue.clone()).unwrap();
    // println!("bandwidth::block_to_global_1d elapsed: {:?}", elapsed);

    // let elapsed = kernels::bandwidth::block_to_global_2d::run(device.clone(), queue.clone()).unwrap();
    // println!("bandwidth::block_to_global_2d elapsed: {:?}", elapsed);

    // let elapsed = kernels::bandwidth::tile_to_global_1d::run(device.clone(), queue.clone()).unwrap();
    // println!("bandwidth::tile_to_global_1d elapsed: {:?}", elapsed);

    // let elapsed = kernels::bandwidth::tile_to_global_2d::run(device.clone(), queue.clone()).unwrap();
    // println!("bandwidth::tile_to_global_2d elapsed: {:?}", elapsed);

    // let elapsed = kernels::bandwidth::vec4_to_global::run(device.clone(), queue.clone()).unwrap();
    // println!("bandwidth::vec4_to_global elapsed: {:?}", elapsed);

    // let elapsed = kernels::bandwidth::tile_to_block_to_global_1d::run(device.clone(), queue.clone()).unwrap();
    // println!("bandwidth::tile_to_block_to_global_1d elapsed: {:?}", elapsed);

    // let elapsed = kernels::matmul::naive::run(device.clone(), queue.clone()).unwrap();
    // println!("matmul::naive elapsed: {:?}", elapsed);

    // let elapsed = kernels::matmul::cache_blocking::run(device.clone(), queue.clone()).unwrap();
    // println!("matmul::cache_blocking elapsed: {:?}", elapsed);

    // let elapsed = kernels::matmul::block_tiling_1d::run(device.clone(), queue.clone()).unwrap();
    // println!("matmul::block_tiling_1d elapsed: {:?}", elapsed);

    // let elapsed = kernels::matmul::block_tiling_2d::run(device.clone(), queue.clone()).unwrap();
    // println!("matmul::block_tiling_2d elapsed: {:?}", elapsed);

    let elapsed = kernels::matmul::vectorize_block_tiling_2d::run(device.clone(), queue.clone()).unwrap();
    println!("matmul::vectorize_block_tiling_2d elapsed: {:?}", elapsed);

    // let elapsed = kernels::matmul::padding::run(device.clone(), queue.clone()).unwrap();
    // println!("matmul::padding elapsed: {:?}", elapsed);

    // let elapsed = kernels::matmul::double_buffer::run(device.clone(), queue.clone()).unwrap();
    // println!("matmul::double_buffer elapsed: {:?}", elapsed);
}
