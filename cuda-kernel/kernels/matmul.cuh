//
// Created by micha on 2023-08-11.
//

#ifndef CUDA_KERNEL_MATMUL_CUH
#define CUDA_KERNEL_MATMUL_CUH

namespace matmul {
    __global__ void hello() {
        uint col = blockIdx.x * blockDim.x + threadIdx.x;
        uint row = blockIdx.y * blockDim.y + threadIdx.y;
        printf("row: %d, col: %d\n", row, col);
    }

    __global__ void naive(const float *matrix_a, const float *matrix_b, float *matrix_c) {
        const uint M = 4096;
        const uint N = 4096;
        const uint K = 4096;

        uint col = blockIdx.x * blockDim.x + threadIdx.x;
        uint row = blockIdx.y * blockDim.y + threadIdx.y;

        float sum = 0.;
        for (int k = 0; k < K; k++) {
            sum += matrix_a[row * K + k] * matrix_b[k * N + col];
        }
        matrix_c[row * N + col] = sum;
    }

    __global__ void cache_blocking(const float *matrix_a, const float *matrix_b, float *matrix_c) {
        const uint M = 4096;
        const uint N = 4096;
        const uint K = 4096;

        // TODO: This can be passed in as a kernel argument
        const uint BM = 32;
        const uint BN = 32;
        const uint BK = 32;

        __shared__ float shared_a[BM * BK];
        __shared__ float shared_b[BK * BN];

        uint global_a_offset = blockIdx.y * BM * K;
        uint global_b_offset = blockIdx.x * BN;
        uint global_c_offset = blockIdx.y * BM * N + blockIdx.x * BN;

        uint thread_index = threadIdx.y * blockDim.x + threadIdx.x;
        uint local_x = thread_index % BN;
        uint local_y = thread_index / BN;

        float sum = 0.0;
        for (uint k = 0; k < K; k += BK) {
            shared_a[local_y * BK + local_x] = matrix_a[global_a_offset + local_y * K + k + local_x];
            shared_b[local_y * BN + local_x] = matrix_b[global_b_offset + (local_y + k) * N + local_x];

            __syncthreads();

            for (uint i = 0; i < BK; i++) {
                sum += shared_a[local_y * BK + i] * shared_b[i * BN + local_x];
            }
            __syncthreads();
        }
        matrix_c[global_c_offset + local_y * N + local_x] = sum;
    }

    /// 2D block tiling matrix multiplication kernel
    __global__ void tiling(const float *matrix_a, const float *matrix_b, float *matrix_c) {
        const uint M = 4096;
        const uint N = 4096;
        const uint K = 4096;

        const uint BM = 32;
        const uint BN = 32;
        const uint BK = 32;

        const uint TM = 8;
        const uint TN = 8;

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
            for (uint i=0; i<BM*BN; i+=num_threads_per_block) {
                uint block_index = i + thread_index;
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

        uint global_c_offset = blockIdx.x * BM * N + blockIdx.y * BN;
        for (uint tile_i=0; tile_i<TM; tile_i++) {
            for (uint tile_j=0; tile_j<TN; tile_j++) {
                uint block_i = thread_x * TM + tile_i;
                uint block_j = thread_y * TN + tile_j;
                if (blockIdx.x*BM + block_i < M && blockIdx.y * BN + block_j < N) {
                    uint global_c_index = global_c_offset + block_i * N + block_j;
                    uint tile_c_index = tile_i * N + tile_j;
                    matrix_c[global_c_index] = tile_c[tile_c_index];
                }
            }
        }
    }
}

#endif //CUDA_KERNEL_MATMUL_CUH
