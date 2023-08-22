
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <iostream>
#include <random>

#include "../cuda-kernel/kernels/common.h"
#include "../cuda-kernel/kernels/matmul.cuh"

using namespace common;

int main()
{
    matmul::hello<<<1, 1>>>();

    float* host_matrix_a, * host_matrix_b, * host_matrix_c;

    host_matrix_a = (float*)malloc(sizeof(float) * M * K);
    host_matrix_b = (float*)malloc(sizeof(float) * K * N);
    host_matrix_c = (float*)malloc(sizeof(float) * M * N);

    generate_random_matrix(host_matrix_a, M, K);
    generate_random_matrix(host_matrix_b, K, N);
    generate_zero_matrix(host_matrix_c, M, N);

    float* device_matrix_a, * device_matrix_b, * device_matrix_c;
    cudaCheck(cudaMalloc((void**)&device_matrix_a, sizeof(float) * M * K));
    cudaCheck(cudaMalloc((void**)&device_matrix_b, sizeof(float) * K * N));
    cudaCheck(cudaMalloc((void**)&device_matrix_c, sizeof(float) * M * N));

    cudaCheck(cudaMemcpy(device_matrix_a, host_matrix_a, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(device_matrix_b, host_matrix_b, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(device_matrix_c, host_matrix_c, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());

    // Use naive implementation as a simple check for now
    float* host_expected, * device_expected;
    host_expected = (float*)malloc(sizeof(float) * M * N);
    cudaCheck(cudaMalloc((void**)&device_expected, sizeof(float) * M * N));

    dim3 block_size(BM, BN, 1);
    dim3 grid_size(M / BM, N / BN, 1);
    matmul::naive<M, N, K><<<grid_size, block_size>>>(alpha, device_matrix_a, device_matrix_b, beta, device_expected);

    cudaCheck(cudaMemcpy(host_expected, device_expected, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    cudaCheck(cudaDeviceSynchronize());

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    cudaCheck(cudaEventRecord(beg));
    unsigned int repeats = 1;
    for (unsigned int i = 0; i < repeats; i++) {

        //block_size = dim3(BM, BN, 1);
        //grid_size = dim3(M / BM, N / BN, 1);
        //matmul::naive<M, N, K><<<grid_size, block_size>>>(alpha, device_matrix_a, device_matrix_b, beta, device_matrix_c);

        //block_size = dim3(BM, BN, 1);
        //grid_size = dim3(M / BM, N / BN, 1);
        //matmul::cache_blocking<M, N, K, BM, BN, BK><<<grid_size, block_size>>>(alpha, device_matrix_a, device_matrix_b, beta, device_matrix_c);

        block_size = dim3(BM / TM, BN, 1);
        grid_size = dim3(M / BM, N / BN, 1);
        matmul::block_tiling_1d<M, N, K, BM, BN, BK, TM><<<grid_size, block_size>>>(alpha, device_matrix_a, device_matrix_b, beta, device_matrix_c);

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

