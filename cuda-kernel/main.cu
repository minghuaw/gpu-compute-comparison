#include <iostream>
#include <random>
#include <chrono>

#include "kernels/matmul.cuh"

/// Generate a contiguous random matrix in row major order
void generate_random_matrix(float *matrix, uint rows, uint cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0, 5.0);

    for (uint i = 0; i < rows * cols; i++) {
        matrix[i] = dis(gen);
    }
}

void generate_zero_matrix(float *matrix, uint rows, uint cols) {
    for (uint i = 0; i < rows * cols; i++) {
        matrix[i] = 0;
    }
}

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
};

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
    cudaCheck(cudaMalloc((void **) &device_matrix_a, sizeof(float) * M * K));
    cudaCheck(cudaMalloc((void **) &device_matrix_b, sizeof(float) * K * N));
    cudaCheck(cudaMalloc((void **) &device_matrix_c, sizeof(float) * M * N));

    cudaCheck(cudaMemcpy(device_matrix_a, host_matrix_a, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(device_matrix_b, host_matrix_b, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(device_matrix_c, host_matrix_c, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    uint BM = 32;
    uint BN = 32;
    dim3 block_size(BM, BN, 1);
    dim3 grid_size(M / BM, N / BN, 1);

    // warm up
//    matmul::naive<<<grid_size, block_size>>>(device_matrix_a, device_matrix_b, device_matrix_c);
//    matmul::naive<<<grid_size, block_size>>>(device_matrix_a, device_matrix_b, device_matrix_c);
//    matmul::cache_blocking<<<grid_size, block_size>>>(device_matrix_a, device_matrix_b, device_matrix_c);
    cudaDeviceSynchronize();

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    cudaEventRecord(beg);
    uint repeats = 1;
    for (uint i = 0; i < repeats; i++) {
//        matmul::naive<<<grid_size, block_size>>>(device_matrix_a, device_matrix_b, device_matrix_c);
        matmul::cache_blocking<<<grid_size, block_size>>>(device_matrix_a, device_matrix_b, device_matrix_c);
    }
    cudaEventRecord(end);

    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);

    printf("Average elapsed time: (%f) ms, performance: (%f) GFLOPS. size: (%d).\n",
           elapsed_time / (float)repeats, 2. * 1e-9 * (float)repeats * M * M * M / elapsed_time, M);

    cudaMemcpy(host_matrix_a, device_matrix_a, sizeof(float) * M * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_matrix_a, device_matrix_a, sizeof(float) * K * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_matrix_c, device_matrix_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    free(host_matrix_a);
    free(host_matrix_b);
    free(host_matrix_c);
    cudaFree(device_matrix_a);
    cudaFree(device_matrix_b);
    cudaFree(device_matrix_c);

    return 0;
}
