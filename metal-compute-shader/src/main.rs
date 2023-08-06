use metal::*;
use objc::rc::autoreleasepool;

fn main() {
    autoreleasepool(|| {
        println!("Hello, world!");
    })
}
