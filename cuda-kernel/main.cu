#include <iostream>
#include <random>
#include <chrono>

#include "kernels/common.h"
#include "kernels/matmul.cuh"
#include "kernels/ported.cuh"

#include "cuda_runtime.h"
#include "cublas_v2.h"

using namespace common;

void cublas_matmul(float *device_matrix_a, float *device_matrix_b, float *device_matrix_c) {
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        printf("Create cublas handle error.\n");
        exit(EXIT_FAILURE);
    };

    //cublas列主序计算：https://www.cnblogs.com/cuancuancuanhao/p/7763256.html
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,N, M, K, &alpha, device_matrix_b, N, device_matrix_a, K, &beta, device_matrix_c, N);

    cublasDestroy(handle);
}

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
        printf("%s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

void assert_eq(float *value, float *expected, uint m, uint n) {
    for (uint i=0; i<m*n; i++) {
        if (value[i] != expected[i]) {
            printf("\x1B[31mError:\033[0m Value: %f is not equal to expected: %f at %d\n", value[i], expected[i], i);
            exit(EXIT_FAILURE);
        }
    }
}

int main() {
    matmul::hello<<<1, 1>>>();

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
    cudaCheck(cudaDeviceSynchronize());

    // Use naive implementation as a simple check for now
    float *host_expected, *device_expected;
    host_expected = (float *)malloc(sizeof(float) * M * N);
    cudaCheck(cudaMalloc((void **)&device_expected, sizeof(float) * M * N));

    dim3 block_size(BM, BN, 1);
    dim3 grid_size(M / BM, N / BN, 1);
//    matmul::naive<<<grid_size, block_size>>>(device_matrix_a, device_matrix_b, device_expected);

    cublas_matmul(device_matrix_a, device_matrix_b, device_expected);
    cudaCheck(cudaMemcpy(host_expected, device_expected, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    cudaCheck(cudaDeviceSynchronize());

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    cudaCheck(cudaEventRecord(beg));
    uint repeats = 1;
    for (uint i = 0; i < repeats; i++) {
//        block_size = dim3(BM, BN, 1);
//        grid_size = dim3(M / BM, N / BN, 1);
//        matmul::naive<<<grid_size, block_size>>>(device_matrix_a, device_matrix_b, device_matrix_c);

//        block_size = dim3(BM, BN, 1);
//        grid_size = dim3(M / BM, N / BN, 1);
//        matmul::cache_blocking<<<grid_size, block_size>>>(device_matrix_a, device_matrix_b, device_matrix_c);

        block_size = dim3(BM * BN / TM, 1, 1);
        grid_size = dim3(M / BM, N / BN, 1);
        ported::block_tiling_1d<<<grid_size, block_size>>>(device_matrix_a, device_matrix_b, device_matrix_c);

//        block_size = dim3(BM / TM, BN / TN, 1);
//        grid_size = dim3(M / BM, N / BN, 1);
//        matmul::tiling<<<grid_size, block_size>>>(device_matrix_a, device_matrix_b, device_matrix_c);
    }
    cudaCheck(cudaEventRecord(end));

    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));

    printf("Average elapsed time: (%f) ms, performance: (%f) GFLOPS. size: (%d).\n",
           elapsed_time / (float)repeats, 2. * 1e-9 * (float)repeats * M * M * M / elapsed_time, M);

    cudaCheck(cudaMemcpy(host_matrix_a, device_matrix_a, sizeof(float) * M * K, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(host_matrix_a, device_matrix_a, sizeof(float) * K * N, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(host_matrix_c, device_matrix_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    cudaCheck(cudaDeviceSynchronize());

    assert_eq(host_matrix_c, host_expected, M, N);

    free(host_matrix_a);
    free(host_matrix_b);
    free(host_matrix_c);
    cudaCheck(cudaFree(device_matrix_a));
    cudaCheck(cudaFree(device_matrix_b));
    cudaCheck(cudaFree(device_matrix_c));

    return 0;
}