#include <iostream>
#include <random>
#include <chrono>

#include "kernels/matmul.cuh"

/// Generate a contiguous random matrix in row major order
void generate_random_matrix(float *matrix, uint rows, uint cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

    for (uint i = 0; i < rows * cols; i++) {
        matrix[i] = dis(gen);
    }
}

void generate_zero_matrix(float *matrix, uint rows, uint cols) {
    for (uint i = 0; i < rows * cols; i++) {
        matrix[i] = 0;
    }
}

int main() {
    matmul::hello<<<1, 1>>>();

    const uint M = 4096;
    const uint N = 4096;
    const uint K = 4096;

    float *host_matrix_a, *host_matrix_b, *host_matrix_c;

    host_matrix_a = (float *)malloc(sizeof(float) * M * K);
    host_matrix_b = (float *)malloc(sizeof(float) * K * N);
    host_matrix_c = (float *)malloc(sizeof(float) * M * N);

    generate_random_matrix(host_matrix_a, M, K);
    generate_random_matrix(host_matrix_b, K, N);
    generate_zero_matrix(host_matrix_c, M, N);

    float *device_matrix_a, *device_matrix_b, *device_matrix_c;
    cudaMalloc((void **) &device_matrix_a, sizeof(float) * M * K);
    cudaMalloc((void **) &device_matrix_b, sizeof(float) * K * N);
    cudaMalloc((void **) &device_matrix_c, sizeof(float) * M * N);

    cudaMemcpy(device_matrix_a, host_matrix_a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix_b, host_matrix_b, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix_c, host_matrix_c, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    dim3 block_size(8, 8, 1);
    dim3 grid_size(M / 8, N / 8, 1);
    matmul::naive<<<block_size, grid_size>>>(device_matrix_a, device_matrix_b, device_matrix_c);
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(s top - start);
    std::cout << duration.count() << std::endl;

    cudaMemcpy(host_matrix_a, device_matrix_a, sizeof(float) * M * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_matrix_a, device_matrix_a, sizeof(float) * K * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_matrix_c, device_matrix_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    free(host_matrix_a);
    free(host_matrix_b);
    free(host_matrix_c);
    cudaFree(device_matrix_a);
    cudaFree(device_matrix_b);
    cudaFree(device_matrix_c);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
