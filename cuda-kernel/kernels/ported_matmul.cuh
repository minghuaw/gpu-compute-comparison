//
// Created by micha on 2023-08-19.
//

#ifndef CUDA_KERNEL_PORTED_MATMUL_CUH
#define CUDA_KERNEL_PORTED_MATMUL_CUH

#include <cassert>

#include "config.h"

namespace ported_matmul {
    using namespace config;

    __global__ void naive(const float *A, const float *B, float *C) {
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = blockIdx.y * blockDim.y + threadIdx.y;

        // if statement is necessary to make things work under tile quantization
        if (x < M && y < N) {
            float tmp = 0.0;
            for (int i = 0; i < K; ++i) {
                tmp += A[x * K + i] * B[i * N + y];
            }
            // C = α*(A@B)+β*C
            C[x * N + y] = alpha * tmp + beta * C[x * N + y];
        }
    }

    __global__ void mem_coalescing(const float *A, const float *B, float *C) {
        const uint cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
        const uint cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

        // if statement is necessary to make things work under tile quantization
        if (cRow < M && cCol < N) {
            float tmp = 0.0;
            for (int i = 0; i < K; ++i) {
                tmp += A[cRow * K + i] * B[i * N + cCol];
            }
            C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
        }
    }

    __global__ void shaded_mem_block(const float *A, const float *B, float *C) {
        // the output block that we want to compute in this threadblock
        const uint cRow = blockIdx.x;
        const uint cCol = blockIdx.y;

        // allocate buffer for current block in fast shared mem
        // shared mem is shared between all threads in a block
        __shared__ float As[BLOCKSIZE * BLOCKSIZE];
        __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

        // the inner row & col that we're accessing in this thread
        const uint threadCol = threadIdx.x % BLOCKSIZE;
        const uint threadRow = threadIdx.x / BLOCKSIZE;

        // advance pointers to the starting positions
        A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
        B += cCol * BLOCKSIZE;                        // row=0, col=cCol
        C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

        float tmp = 0.0;
        for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
            // Have each thread load one of the elements in A & B
            // Make the threadCol (=threadIdx.x) the consecutive index
            // to allow global memory access coalescing
            As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
            Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

            // block threads in this block until cache is fully populated
            __syncthreads();
            A += BLOCKSIZE;
            B += BLOCKSIZE * N;

            // execute the dotproduct on the currently cached block
            for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
                tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                       Bs[dotIdx * BLOCKSIZE + threadCol];
            }
            // need to sync again at the end, to avoid faster threads
            // fetching the next block into the cache before slower threads are done
            __syncthreads();
        }
        C[threadRow * N + threadCol] =
                alpha * tmp + beta * C[threadRow * N + threadCol];
    }

    __global__ void block_tiling_1d(const float *A, const float *B, float *C) {
        // If we flip x and y here we get ~30% less performance for large matrices.
        // The current, 30% faster configuration ensures that blocks with sequential
        // blockIDs access columns of B sequentially, while sharing the same row of A.
        // The slower configuration would share columns of A, but access into B would
        // be non-sequential. So the faster configuration has better spatial locality
        // and hence a greater L2 hit rate.
        const uint cRow = blockIdx.y;
        const uint cCol = blockIdx.x;

        // each warp will calculate 32*TM elements, with 32 being the columnar dim.
        const int threadCol = threadIdx.x % BN;
        const int threadRow = threadIdx.x / BN;

        // allocate space for the current blocktile in SMEM
        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];

        // Move blocktile to beginning of A's row and B's column
        A += cRow * BM * K;
        B += cCol * BN;
        C += cRow * BM * N + cCol * BN;

        // todo: adjust this to each thread to load multiple entries and
        // better exploit the cache sizes
//        assert(BM * BK == blockDim.x);
//        assert(BN * BK == blockDim.x);
        const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
        const uint innerRowA = threadIdx.x / BK;
        const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
        const uint innerRowB = threadIdx.x / BN;

        // allocate thread-local cache for results in registerfile
        float threadResults[TM] = {0.0};

        // outer loop over block tiles
        for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
            // populate the SMEM caches
            As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
            Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
            __syncthreads();

            // advance blocktile
            A += BK;
            B += BK * N;

            // calculate per-thread results
            for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                // we make the dotproduct loop the outside loop, which facilitates
                // reuse of the Bs entry, which we can cache in a tmp var.
                float tmpB = Bs[dotIdx * BN + threadCol];
                for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                    threadResults[resIdx] +=
                            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
                }
            }
            __syncthreads();
        }

        // write out the results
        for (uint resIdx = 0; resIdx < TM; ++resIdx) {
            C[(threadRow * TM + resIdx) * N + threadCol] =
                    alpha * threadResults[resIdx] +
                    beta * C[(threadRow * TM + resIdx) * N + threadCol];
        }
    }

    __global__ void block_tiling_2d(const float *A, const float *B, float *C) {
        const uint cRow = blockIdx.y;
        const uint cCol = blockIdx.x;

        const uint totalResultsBlocktile = BM * BN;
        // A thread is responsible for calculating TM*TN elements in the blocktile
        const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

        // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
        assert(numThreadsBlocktile == blockDim.x);

        // BN/TN are the number of threads to span a column
        const int threadCol = threadIdx.x % (BN / TN);
        const int threadRow = threadIdx.x / (BN / TN);

        // allocate space for the current blocktile in smem
        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];

        // Move blocktile to beginning of A's row and B's column
        A += cRow * BM * K;
        B += cCol * BN;
        C += cRow * BM * N + cCol * BN;

        // calculating the indices that this thread will load into SMEM
        const uint innerRowA = threadIdx.x / BK;
        const uint innerColA = threadIdx.x % BK;
        // calculates the number of rows of As that are being loaded in a single step
        // by a single block
        const uint strideA = numThreadsBlocktile / BK;
        const uint innerRowB = threadIdx.x / BN;
        const uint innerColB = threadIdx.x % BN;
        // for both As and Bs we want each load to span the full column-width, for
        // better GMEM coalescing (as opposed to spanning full row-width and iterating
        // across columns)
        const uint strideB = numThreadsBlocktile / BN;

        // allocate thread-local cache for results in registerfile
        float threadResults[TM * TN] = {0.0};
        // register caches for As and Bs
        float regM[TM] = {0.0};
        float regN[TN] = {0.0};

        // outer-most loop over block tiles
        for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
            // populate the SMEM caches
            for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
                As[(innerRowA + loadOffset) * BK + innerColA] =
                        A[(innerRowA + loadOffset) * K + innerColA];
            }
            for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
                Bs[(innerRowB + loadOffset) * BN + innerColB] =
                        B[(innerRowB + loadOffset) * N + innerColB];
            }
            __syncthreads();

            // advance blocktile
            A += BK;     // move BK columns to right
            B += BK * N; // move BK rows down

            // calculate per-thread results
            for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                // block into registers
                for (uint i = 0; i < TM; ++i) {
                    regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
                }
                for (uint i = 0; i < TN; ++i) {
                    regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
                }
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        threadResults[resIdxM * TN + resIdxN] +=
                                regM[resIdxM] * regN[resIdxN];
                    }
                }
            }
            __syncthreads();
        }

        // write out the results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                        alpha * threadResults[resIdxM * TN + resIdxN] +
                        beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
            }
        }
    }
}


#endif //CUDA_KERNEL_PORTED_MATMUL_CUH
