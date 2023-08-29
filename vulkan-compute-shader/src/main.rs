use kernels::bandwidth;

extern crate openblas_src;

mod common;
mod kernels;

fn main() {
    let (device, queue) = common::get_device_and_queue().unwrap();

    let elapsed = bandwidth::block_1d_to_global::run(device.clone(), queue.clone()).unwrap();
    println!("bandwidth::block_1d_to_global elapsed: {:?}", elapsed);

    let elapsed = bandwidth::block_2d_to_global::run(device.clone(), queue.clone()).unwrap();
    println!("bandwidth::block_2d_to_global elapsed: {:?}", elapsed);

    let elapsed = bandwidth::tile_1d_to_global::run(device.clone(), queue.clone()).unwrap();
    println!("bandwidth::tile_1d_to_global elapsed: {:?}", elapsed);

    // let elapsed = matmul::naive::run(device.clone(), queue.clone()).unwrap();
    // println!("matmul::naive elapsed: {:?}", elapsed);
}
