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
}

#endif //CUDA_KERNEL_MATMUL_CUH
