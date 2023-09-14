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

//        block_size = dim3(BM / TM, BN, 1);
//        grid_size = dim3(M / BM, N / BN, 1);
//        matmul::block_tiling_1d<M, N, K, BM, BN, BK, TM><<<grid_size, block_size>>>(alpha, device_matrix_a, device_matrix_b, beta, device_matrix_c);

//        // A 2d block will give wrong result somehow
//        block_size = dim3(64, 1, 1);
//        grid_size = dim3(M / 64, N / 64, 1);
//        matmul::block_tiling_2d<M, N, K, 64, 64, 8, 8, 8><<<grid_size, block_size>>>(alpha, device_matrix_a, device_matrix_b, beta, device_matrix_c);

//        block_size = dim3(256, 1, 1);
//        grid_size = dim3(M / 128, N / 128, 1);
//        matmul::vectorize_block_tiling_2d<128, 128, 8, 8, 8><<<grid_size, block_size>>>(M, N, K, alpha, device_matrix_a, device_matrix_b, beta, device_matrix_c);

//        const uint K10_NUM_THREADS = 128;
//        const uint K10_BN = 128;
//        const uint K10_BM = 128;
//        const uint K10_BK = 16;
//        const uint K10_WN = 64;
//        const uint K10_WM = 64;
//        const uint K10_WNITER = 4;
//        const uint K10_TN = 4;
//        const uint K10_TM = 8;
//        dim3 blockDim(K10_NUM_THREADS);
//
//        constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;
//
//        // warptile in threadblocktile
//        static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
//        static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);
//
//        // threads in warpsubtile
//        static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
//                      0);
//        constexpr uint K10_WMITER =
//                (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
//        // warpsubtile in warptile
//        static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));
//
//        static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
//                      "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
//                      "issues during GMEM->SMEM tiling (loading only parts of the "
//                      "final row of Bs during each iteraion)");
//        static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
//                      "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
//                      "issues during GMEM->SMEM tiling (loading only parts of the "
//                      "final row of As during each iteration)");
//        static_assert(K10_BN % (16 * K10_TN) == 0,
//                      "BN must be a multiple of 16*TN to avoid quantization effects");
//        static_assert(K10_BM % (16 * K10_TM) == 0,
//                      "BM must be a multiple of 16*TM to avoid quantization effects");
//        static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
//                      "BM*BK must be a multiple of 4*256 to vectorize loads");
//        static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
//                      "BN*BK must be a multiple of 4*256 to vectorize loads");
//
//        dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
//        ported::sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
//                K10_TN, K10_NUM_THREADS>
//        <<<gridDim, blockDim>>>(M, N, K, alpha, device_matrix_a, device_matrix_b, beta, device_matrix_c);

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