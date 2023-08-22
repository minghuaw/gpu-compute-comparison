//
// Created by micha on 2023-08-21.
//

#ifndef CUDA_KERNEL_PORTED_CUH
#define CUDA_KERNEL_PORTED_CUH

#include "common.h"

namespace ported {
    using namespace common;

    // TODO: The result is incorrect
    __global__ void block_tiling_1d(const float *matrix_a, const float *matrix_b, float *matrix_c) {
        uint thread_num = BM * BN / TM;

        __shared__ float shared_a[BM * BK];
        __shared__ float shared_b[BK * BN];

        uint thread_index = threadIdx.y * blockDim.x + threadIdx.x;

        uint tx = thread_index % BN;
        uint ty = thread_index / BN * TM;

        uint a_global_offset = blockIdx.y * BM * K;
        uint b_global_offset = blockIdx.x * BN;
        const uint c_global_offset = blockIdx.y * BM * N + blockIdx.x * BN;

        uint a_tile_row = thread_index / BK;
        uint a_tile_col = thread_index % BK;
        uint a_tile_stride = thread_num / BK;

        uint b_tile_row = thread_index / BN;
        uint b_tile_col = thread_index % BN;
        uint b_tile_stride = thread_num / BN;

        float c_tile[TM];
        float b_cache;
        for (uint k=0; k < K; k += BK) {
            for (uint i=0; i<BM; i += a_tile_stride) {
                shared_a[(a_tile_row + i) * BK + a_tile_col] = matrix_a[a_global_offset + (a_tile_row + i) * K + a_tile_col];
            }
            for (uint i=0; i<BK; i += b_tile_stride) {
                shared_b[(b_tile_row + i) * BN + b_tile_col] = matrix_b[b_global_offset + (b_tile_row + i) * N + b_tile_col];
            }

            __syncthreads();
            a_global_offset += BK;
            b_global_offset += BK * N;

            for (uint i=0; i<BK; i++) {
                b_cache = shared_b[tx + i * BN];
                for (uint j=0; j<TM; j++) {
                    c_tile[j] += shared_a[(ty + j) * BK + i] * b_cache;
                }
            }
            __syncthreads();
        }
        for (uint j=0; j<TM; j++) {
            matrix_c[c_global_offset + (ty + j) * N + tx] = alpha * c_tile[j] + beta * matrix_c[c_global_offset + (ty + j) * N + tx];
        }
    }
}

#endif //CUDA_KERNEL_PORTED_CUH
