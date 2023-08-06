use std::process::Command;

fn main() {
    Command::new("echo")
        .args(&["Hello, build script!"])
        .status()
        .unwrap();
    Command::new("xcrun")
        .args(&["-sdk", "macosx", "metal", "shaders/matmul.metal", "-o", "shaders/matmul.air"])
        .status()
        .unwrap();
    Command::new("xcrun")
        .args(&["-sdk", "macosx", "metallib", "shaders/matmul.air", "-o", "shaders/matmul.metallib"])
        .status()
        .unwrap();
    println!("cargo:rerun-if-changed=shaders/matmul.metal");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.lock")
}