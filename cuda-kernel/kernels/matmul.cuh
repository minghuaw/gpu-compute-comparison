//
// Created by micha on 2023-08-11.
//

#ifndef CUDA_KERNEL_MATMUL_CUH
#define CUDA_KERNEL_MATMUL_CUH

#include "config.h"

namespace matmul {
    using namespace config;
    __global__ void hello() {
        printf("threadIdx.x: %d\n", threadIdx.x);
    }

    __global__ void naive(const float *matrix_a, const float *matrix_b, float *matrix_c) {
        // Having col on x and row on y is critical
        uint col = blockIdx.x * blockDim.x + threadIdx.x;
        uint row = blockIdx.y * blockDim.y + threadIdx.y;

        float sum = 0.;
        for (int k = 0; k < K; k++) {
            sum += matrix_a[row * K + k] * matrix_b[k * N + col];
        }
        matrix_c[row * N + col] = alpha * sum + beta * matrix_c[row * N + col];
    }

    __global__ void cache_blocking(const float *matrix_a, const float *matrix_b, float *matrix_c) {
        __shared__ float shared_a[BM * BK];
        __shared__ float shared_b[BK * BN];

        uint global_a_offset = blockIdx.y * BM * K;
        uint global_b_offset = blockIdx.x * BN;
        uint global_c_offset = blockIdx.y * BM * N + blockIdx.x * BN;

        uint thread_index = threadIdx.y * blockDim.x + threadIdx.x;
        uint thread_col = thread_index % BN;
        uint thread_row = thread_index / BN;

        float sum = 0.0;
        for (uint k = 0; k < K; k += BK) {
            shared_a[thread_row * BK + thread_col] = matrix_a[global_a_offset + thread_row * K + k + thread_col];
            shared_b[thread_row * BN + thread_col] = matrix_b[global_b_offset + (thread_row + k) * N + thread_col];

            __syncthreads();

            for (uint i = 0; i < BK; i++) {
                sum += shared_a[thread_row * BK + i] * shared_b[i * BN + thread_col];
            }
            __syncthreads();
        }
        matrix_c[global_c_offset + thread_row * N + thread_col] = alpha * sum + beta
                * matrix_c[global_c_offset + thread_row * N + thread_col];
    }

    /// 2D block tiling matrix multiplication kernel
    __global__ void tiling(const float *matrix_a, const float *matrix_b, float *matrix_c) {
        __shared__ float shared_a[BM * BK];
        __shared__ float shared_b[BK * BN];

        uint num_threads_per_block = (BM * BN) / (TM * TN);

        uint thread_index = threadIdx.y * blockDim.x + threadIdx.x;
        uint thread_x = thread_index / (BN / TN);
        uint thread_y = thread_index % (BN / TN);

        float tile_a[TM];
        float tile_b[TN];
        float tile_c[TM * TN];

        for (uint global_k=0; global_k<BK; global_k+=BK) {
            for (uint i=0; i<BM*BK; i+=num_threads_per_block) {
                uint block_index = i + thread_index;
                if (block_index < BM*BK) {
                    uint block_i = block_index / BK;
                    uint block_j = block_index % BK;
                    uint global_i = blockIdx.x * BM + block_i;
                    uint global_j = global_k + block_j;
                    if (global_i < M && global_j < K) {
                        shared_a[block_index] = matrix_a[global_i * K + global_j];
                    } else {
                        shared_a[block_index] = 0;
                    }
                }
            }
            for (uint i=0; i<BK*BN; i+=num_threads_per_block) {
                uint block_index = i + thread_index;
                if (block_index < BK * BN) {
                    uint block_i = block_index / BN;
                    uint block_j = block_index % BN;
                    uint global_i = global_k + block_i;
                    uint global_j = blockIdx.y * BN + block_j;
                    if (global_i < K && global_j < N) {
                        shared_b[block_index] = matrix_b[global_i * N + global_j];
                    } else {
                        shared_b[block_index] = 0;
                    }
                }
            }
            __syncthreads();

            for (uint block_k=0; block_k<BK; block_k++) {
                for (uint tile_i=0; tile_i<TM; tile_i++) {
                    tile_a[tile_i] = shared_a[(thread_x*TM + tile_i) * BK + block_k];
                }
                for (uint tile_j=0; tile_j<TN; tile_j++) {
                    tile_b[tile_j] = shared_b[block_k * BN + (thread_y * TN + tile_j)];
                }

                for (uint tile_i=0; tile_i < TM; tile_i++) {
                    for (uint tile_j=0;  tile_j<TN; tile_j++) {
                        tile_c[tile_i * TM + tile_j] += tile_a[tile_i] * tile_b[tile_j];
                    }
                }
            }
            __syncthreads();
        }

        for (uint tile_i=0; tile_i<TM; tile_i++) {
            for (uint tile_j=0; tile_j<TN; tile_j++) {
                uint block_i = thread_x * TM + tile_i;
                uint block_j = thread_y * TN + tile_j;
                uint global_c_offset = blockIdx.x * BM * N + blockIdx.y * BN;
                uint global_c_index = global_c_offset + block_i * N + block_j;
                uint tile_c_index = tile_i * TM + tile_j;
                if (global_c_index < M*N && tile_c_index < TM*TN) {
                    matrix_c[global_c_index] = tile_c[tile_c_index];
                }
            }
        }
    }
}

#endif //CUDA_KERNEL_MATMUL_CUH
