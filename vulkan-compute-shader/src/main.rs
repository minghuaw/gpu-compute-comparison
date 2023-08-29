use kernels::bandwidth;

extern crate openblas_src;

mod common;
mod kernels;

fn main() {
    let (device, queue) = common::get_device_and_queue().unwrap();

    let elapsed = bandwidth::block_to_global_1d::run(device.clone(), queue.clone()).unwrap();
    println!("bandwidth::block_to_global_1d elapsed: {:?}", elapsed);

    let elapsed = bandwidth::block_to_global_2d::run(device.clone(), queue.clone()).unwrap();
    println!("bandwidth::block_to_global_2d elapsed: {:?}", elapsed);

    let elapsed = bandwidth::tile_to_global_1d::run(device.clone(), queue.clone()).unwrap();
    println!("bandwidth::tile_to_global_1d elapsed: {:?}", elapsed);

    let elapsed = bandwidth::block_tile_to_global_1d::run(device.clone(), queue.clone()).unwrap();
    println!("bandwidth::block_tile_to_global_1d elapsed: {:?}", elapsed);

    // let elapsed = matmul::naive::run(device.clone(), queue.clone()).unwrap();
    // println!("matmul::naive elapsed: {:?}", elapsed);
}
