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

            float sum = 0.0f;
            for (uint i = 0; i < BK; i++) {
                sum += shared_a[local_y * BK + i] * shared_b[i * BN + local_x];
            }
            __syncthreads();
        }
        matrix_c[global_c_offset + local_y * N + local_x] = sum;
    }
}

#endif //CUDA_KERNEL_MATMUL_CUH
