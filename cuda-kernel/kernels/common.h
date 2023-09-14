//
// Created by micha on 2023-08-21.
//

#ifndef CUDA_KERNEL_COMMON_H
#define CUDA_KERNEL_COMMON_H

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

namespace common {
    const unsigned int M = 4096;
    const unsigned int N = 4096;
    const unsigned int K = 4096;

    // TODO: This can be passed in as a kernel argument
    // Keep these three the same for now. There is likely bug in indexing that will cause illegal access
    const unsigned int BM = 32;
    const unsigned int BN = 32;
    const unsigned int BK = 32;

    const unsigned int TM = 8;
    const unsigned int TN = 8;

    const float alpha = 1.0;
    const float beta = 1.0;

    const int WARPSIZE = 32; // warpSize is not constexpr

    /// Generate a contiguous random matrix in row major order
    void generate_random_matrix(float *matrix, unsigned int rows, unsigned int cols) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(1.0, 5.0);

        for (unsigned int i = 0; i < rows * cols; i++) {
            matrix[i] = dis(gen);
        }
    }

    void generate_zero_matrix(float *matrix, unsigned int rows, unsigned int cols) {
        for (unsigned int i = 0; i < rows * cols; i++) {
            matrix[i] = 0;
        }
    }

    void cudaCheck(cudaError_t error) {
        if (error != cudaSuccess) {
            printf("%s", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    };

    void assert_eq(float *value, float *expected, unsigned int m, unsigned int n) {
        for (unsigned int i = 0; i < m * n; i++) {
            if (value[i] != expected[i]) {
                printf("\x1B[31mError:\033[0m Value: %f is not equal to expected: %f at %d\n", value[i], expected[i],
                       i);
                exit(EXIT_FAILURE);
            }
        }
    }
}

#endif //CUDA_KERNEL_COMMON_H
