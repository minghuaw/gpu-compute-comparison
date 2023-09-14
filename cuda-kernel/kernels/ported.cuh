//
// Created by micha on 2023-08-21.
//

#ifndef CUDA_KERNEL_PORTED_CUH
#define CUDA_KERNEL_PORTED_CUH

#include "common.h"

namespace ported {
    using namespace common;

    //__global__ void block_tiling_1d(const float *A, const float *B, float *C) {
    //    int bx = blockIdx.x;
    //    int by = blockIdx.y;
    //    int thread_num = BM * BN / TM; // 一个线程负责block中计算TM个元素

    //    int tx = threadIdx.x % BN;
    //    int ty = threadIdx.x / BN * TM;

    //    __shared__ float As[BM * BK];
    //    __shared__ float Bs[BK * BN];

    //    // 移动到当前block
    //    A = &A[by * BM * K];
    //    B = &B[bx * BN];
    //    C = &C[by * BM * N + bx * BN];

    //    /*
    //    当前线程负责搬运全局内存中第a_tile_row行，第a_tile_col列元素至共享内存第a_tile_row行，第a_tile_col列
    //    a_tile_stride表示block中线程可搬运a_tile_stride行至共享内存；

    //    若BM=64,BK=8,thread_num=512,则a_tile_stride=64,a_tile_stride=BM，表示每个线程搬运一轮即可完成所需元素的搬运;
    //    若BM=128,BK=8,thread_num=512,则a_tile_stride=64,表示每个线程搬运两轮即可完成所需元素的搬运;
    //    */
    //    int a_tile_row = threadIdx.x / BK;
    //    int a_tile_col = threadIdx.x % BK;
    //    int a_tile_stride = thread_num / BK;

    //    int b_tile_row = threadIdx.x / BN;
    //    int b_tile_col = threadIdx.x % BN;
    //    int b_tile_stride = thread_num / BN;

    //    float tmp[TM + 1] = {0.}; // 每个线程负责TM个元素，则需要申请TM个寄存器保存累加值，额外的一个寄存器用于缓存；
    //    for (int k = 0; k < K; k += BK) {
    //        for (int i = 0; i < BM; i += a_tile_stride) {
    //            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
    //        }
    //        for (int i = 0; i < BK; i += b_tile_stride) {
    //            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
    //        }
    //        __syncthreads();
    //        A += BK;
    //        B += BK * N;
    //        for (int i = 0; i < BK; i++) {
    //            tmp[TM] = Bs[tx + i * BN]; // 额外的一个寄存器，避免反复从共享内存中读取Bs[tx + i * BN]
    //            for (int j = 0; j < TM; j++) {
    //                tmp[j] += As[(ty + j) * BK + i] * tmp[TM];
    //            }
    //        }
    //        __syncthreads();
    //    }
    //    for (int j = 0; j < TM; j++) {
    //        C[(ty + j) * N + tx] = alpha * tmp[j] + beta * C[(ty + j) * N + tx];
    //    }
    //}

    template<
            const unsigned int M,
            const unsigned int N,
            const unsigned int K,
            const unsigned int BM,
            const unsigned int BN,
            const unsigned int BK,
            const unsigned int TM,
            const unsigned int TN
    >
    __global__ void block_tiling_2d(
            const float alpha,
            const float *A,
            const float *B,
            const float beta,
            float *C
    ) {
        int bx = blockIdx.x;
        int by = blockIdx.y;

        int block_row_thread = BN / TN;
        int block_col_thread = BM / TM;
        int thread_num = block_row_thread * block_col_thread; // 一个线程负责计算block中TM*TN个元素

        int tx = (threadIdx.x % block_row_thread) * TN;
        int ty = (threadIdx.x / block_row_thread) * TM;

        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];

        // 移动到当前block
        A = &A[by * BM * K];
        B = &B[bx * BN];
        C = &C[by * BM * N + bx * BN];

        /*
        当前线程负责搬运全局内存中第a_tile_row行，第a_tile_col列元素至共享内存第a_tile_row行，第a_tile_col列
        a_tile_stride表示block中线程可搬运a_tile_stride行至共享内存；

        若BM=64,BK=8,thread_num=512,则a_tile_stride=64,a_tile_stride=BM，表示每个线程搬运一轮即可完成所需元素的搬运;
        若BM=128,BK=8,thread_num=512,则a_tile_stride=64,表示每个线程搬运两轮即可完成所需元素的搬运;
        */
        int a_tile_row = threadIdx.x / BK;
        int a_tile_col = threadIdx.x % BK;
        int a_tile_stride = thread_num / BK;

        int b_tile_row = threadIdx.x / BN;
        int b_tile_col = threadIdx.x % BN;
        int b_tile_stride = thread_num / BN;

        float tmp[TM][TN] = {0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；
#pragma unroll
        for (int k = 0; k < K; k += BK) {
#pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
            }
            __syncthreads();
            A += BK;
            B += BK * N;
#pragma unroll
            for (int i = 0; i < BK; i++) {
#pragma unroll  // 循环展开，增加指令并行度
                for (int j = 0; j < TM; j++) {
                    for (int l = 0; l < TN; l++)
                        tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < TM; j++) {
            for (int l = 0; l < TN; l++)
                C[(ty + j) * N + tx + l] = alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
        }
    }

    namespace wt {
        template <const int BM, const int BN, const int BK, const int rowStrideA,
                const int rowStrideB>
        __device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                                     float *As, float *Bs, int innerRowA, int innerColA,
                                     int innerRowB, int innerColB) {
            for (unsigned int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
                const float4 tmp = reinterpret_cast<const float4 *>(
                        &A[(innerRowA + offset) * K + innerColA * 4])[0];
                // float4 tmp;
                // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
                //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
                As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
                As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
                As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
                As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
            }

            for (unsigned int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
                reinterpret_cast<float4 *>(
                        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                        reinterpret_cast<const float4 *>(
                                &B[(innerRowB + offset) * N + innerColB * 4])[0];
                // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
                //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
                //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
                //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
                //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
            }
        }

        template <const int BM, const int BN, const int BK, const int WM, const int WN,
                const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
                const int TM, const int TN>
        __device__ void
        processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                        const float *Bs, const unsigned int warpRow, const unsigned int warpCol,
                        const unsigned int threadRowInWarp, const unsigned int threadColInWarp) {
            for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
                // populate registers for whole warptile
                for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    for (unsigned int i = 0; i < TM; ++i) {
                        regM[wSubRowIdx * TM + i] =
                                As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                                   threadRowInWarp * TM + i];
                    }
                }
                for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    for (unsigned int i = 0; i < TN; ++i) {
                        regN[wSubColIdx * TN + i] =
                                Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                                   threadColInWarp * TN + i];
                    }
                }

                // execute warptile matmul
                for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                        // calculate per-thread results
                        for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                            for (unsigned int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                                threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                              (wSubColIdx * TN) + resIdxN] +=
                                        regM[wSubRowIdx * TM + resIdxM] *
                                        regN[wSubColIdx * TN + resIdxN];
                            }
                        }
                    }
                }
            }
        }
    }

    /*
     * @tparam BM The threadblock size for M dimension SMEM caching.
     * @tparam BN The threadblock size for N dimension SMEM caching.
     * @tparam BK The threadblock size for K dimension SMEM caching.
     * @tparam WM M dim of continuous tile computed by each warp
     * @tparam WN N dim of continuous tile computed by each warp
     * @tparam WMITER The number of subwarp tiling steps in M dimension.
     * @tparam WNITER The number of subwarp tiling steps in N dimension.
     * @tparam TM The per-thread tile size for M dimension.
     * @tparam TN The per-thread tile size for N dimension.
     */
    template <const int BM, const int BN, const int BK, const int WM, const int WN,
            const int WNITER, const int TM, const int TN, const int NUM_THREADS>
    __global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
        const unsigned int cRow = blockIdx.y;
        const unsigned int cCol = blockIdx.x;

        // Placement of the warp in the threadblock tile
        const unsigned int warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
        const unsigned int warpCol = warpIdx % (BN / WN);
        const unsigned int warpRow = warpIdx / (BN / WN);

        // size of the warp subtile
        constexpr unsigned int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
        constexpr unsigned int WSUBM = WM / WMITER; // 64/2=32
        constexpr unsigned int WSUBN = WN / WNITER; // 32/2=16

        // Placement of the thread in the warp subtile
        const unsigned int threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
        const unsigned int threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
        const unsigned int threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

        // allocate space for the current blocktile in SMEM
        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];

        // Move blocktile to beginning of A's row and B's column
        A += cRow * BM * K;
        B += cCol * BN;
        // Move C_ptr to warp's output tile
        C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

        // calculating the indices that this thread will load into SMEM
        // we'll load 128bit / 32bit = 4 elements per thread at each step
        const unsigned int innerRowA = threadIdx.x / (BK / 4);
        const unsigned int innerColA = threadIdx.x % (BK / 4);
        constexpr unsigned int rowStrideA = (NUM_THREADS * 4) / BK;
        const unsigned int innerRowB = threadIdx.x / (BN / 4);
        const unsigned int innerColB = threadIdx.x % (BN / 4);
        constexpr unsigned int rowStrideB = NUM_THREADS / (BN / 4);

        // allocate thread-local cache for results in registerfile
        float threadResults[WMITER * TM * WNITER * TN] = {0.0};
        // we cache into registers on the warptile level
        float regM[WMITER * TM] = {0.0};
        float regN[WNITER * TN] = {0.0};

        // outer-most loop over block tiles
        for (unsigned int bkIdx = 0; bkIdx < K; bkIdx += BK) {
            wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
                    N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
            __syncthreads();
            wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                    TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                        threadRowInWarp, threadColInWarp);
            A += BK;     // move BK columns to right
            B += BK * N; // move BK rows down
            __syncthreads();
        }

        // write out the results
        for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                // move C pointer to current warp subtile
                float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
                for (unsigned int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                    for (unsigned int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                        // load C vector into registers
                        float4 tmp = reinterpret_cast<float4 *>(
                                &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                           threadColInWarp * TN + resIdxN])[0];
                        // perform GEMM update in reg
                        const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                      wSubColIdx * TN + resIdxN;
                        tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                        tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                        tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                        tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                        // write back
                        reinterpret_cast<float4 *>(
                                &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                           threadColInWarp * TN + resIdxN])[0] = tmp;
                    }
                }
            }
        }
    }
}

#endif //CUDA_KERNEL_PORTED_CUH
