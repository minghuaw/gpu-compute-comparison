//
// Created by micha on 2023-08-21.
//

#ifndef CUDA_KERNEL_PORTED_CUH
#define CUDA_KERNEL_PORTED_CUH

#include "common.h"

namespace ported {
    using namespace common;

    __global__ void block_tiling_1d(const float *A, const float *B, float *C) {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int thread_num = BM * BN / TM; // 一个线程负责block中计算TM个元素

        int tx = threadIdx.x % BN;
        int ty = threadIdx.x / BN * TM;

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

        float tmp[TM + 1] = {0.}; // 每个线程负责TM个元素，则需要申请TM个寄存器保存累加值，额外的一个寄存器用于缓存；
        for (int k = 0; k < K; k += BK) {
            for (int i = 0; i < BM; i += a_tile_stride) {
                As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
            }
            for (int i = 0; i < BK; i += b_tile_stride) {
                Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
            }
            __syncthreads();
            A += BK;
            B += BK * N;
            for (int i = 0; i < BK; i++) {
                tmp[TM] = Bs[tx + i * BN]; // 额外的一个寄存器，避免反复从共享内存中读取Bs[tx + i * BN]
                for (int j = 0; j < TM; j++) {
                    tmp[j] += As[(ty + j) * BK + i] * tmp[TM];
                }
            }
            __syncthreads();
        }
        for (int j = 0; j < TM; j++) {
            C[(ty + j) * N + tx] = alpha * tmp[j] + beta * C[(ty + j) * N + tx];
        }
    }

//    __global__ void block_tiling_1d(const float *A, const float *B, float *C) {
//        __shared__ float As[BM * BK];
//        __shared__ float Bs[BK * BN];
//
//        uint bx = blockIdx.x;
//        uint by = blockIdx.y;
//
//        uint thread_index = threadIdx.y * blockDim.x + threadIdx.x;
//
//        uint thread_num = BM * BN / TM; // 一个线程负责block中计算TM个元素
//
//        uint tx = thread_index % BN;
//        uint ty = thread_index / BN * TM;
//
//        uint a_global_offset = by * BM * K;
//        uint b_global_offset = bx * BN;
//        uint c_global_offset = by * BM * N + bx * BN;
//
//        /*
//        当前线程负责搬运全局内存中第a_tile_row行，第a_tile_col列元素至共享内存第a_tile_row行，第a_tile_col列
//        a_tile_stride表示block中线程可搬运a_tile_stride行至共享内存；
//
//        若BM=64,BK=8,thread_num=512,则a_tile_stride=64,a_tile_stride=BM，表示每个线程搬运一轮即可完成所需元素的搬运;
//        若BM=128,BK=8,thread_num=512,则a_tile_stride=64,表示每个线程搬运两轮即可完成所需元素的搬运;
//        */
//        uint a_tile_row = thread_index / BK;
//        uint a_tile_col = thread_index % BK;
//        uint a_tile_stride = thread_num / BK;
//
//        uint b_tile_row = thread_index / BN;
//        uint b_tile_col = thread_index % BN;
//        uint b_tile_stride = thread_num / BN;
//
//        float tmp[TM] = {0.}; // 每个线程负责TM个元素，则需要申请TM个寄存器保存累加值，额外的一个寄存器用于缓存；
//        float b_cache = 0.0;
//        for (uint k = 0; k < K; k += BK) {
//            for (uint i = 0; i < BM; i += a_tile_stride) {
//                As[(a_tile_row + i) * BK + a_tile_col] = A[a_global_offset + (a_tile_row + i) * K + a_tile_col];
//            }
//            for (uint i = 0; i < BK; i += b_tile_stride) {
//                Bs[(b_tile_row + i) * BN + b_tile_col] = B[b_global_offset + (b_tile_row + i) * N + b_tile_col];
//            }
//            __syncthreads();
//            a_global_offset += BK;
//            b_global_offset += BK * N;
//            for (uint i = 0; i < BK; i++) {
//                b_cache = Bs[tx + i * BN]; // 额外的一个寄存器，避免反复从共享内存中读取Bs[tx + i * BN]
//                for (uint j = 0; j < TM; j++) {
//                    tmp[j] += As[(ty + j) * BK + i] * b_cache;
//                }
//            }
//            __syncthreads();
//        }
//        for (int j = 0; j < TM; j++) {
//            C[c_global_offset + (ty + j) * N + tx] = alpha * tmp[j] + beta * C[c_global_offset + (ty + j) * N + tx];
//        }
//    }
}

#endif //CUDA_KERNEL_PORTED_CUH
