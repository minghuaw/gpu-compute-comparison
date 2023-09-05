//
// Created by micha on 2023-08-11.
//

#ifndef CUDA_KERNEL_MATMUL_CUH
#define CUDA_KERNEL_MATMUL_CUH

namespace matmul {
    __global__ void hello() {
        unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        printf("row: %d, col: %d\n", row, col);
    }

    template<
            const unsigned int M,
            const unsigned int N,
            const unsigned int K
    >
    __global__ void naive(
            const float alpha,
            const float *matrix_a,
            const float *matrix_b,
            const float beta,
            float *matrix_c
    ) {
        unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

        float sum = 0.;
        for (int k = 0; k < K; k++) {
            sum += matrix_a[row * K + k] * matrix_b[k * N + col];
        }
        matrix_c[row * N + col] = alpha * sum + beta * matrix_c[row * N + col];
    }

    template<
            const unsigned int M,
            const unsigned int N,
            const unsigned int K,
            const unsigned int BM,
            const unsigned int BN,
            const unsigned int BK
    >
    __global__ void cache_blocking(
            const float alpha,
            const float *matrix_a,
            const float *matrix_b,
            const float beta,
            float *matrix_c
    ) {
        // TODO: This can be passed in as a kernel argument
        __shared__ float shared_a[BM * BK];
        __shared__ float shared_b[BK * BN];

        unsigned int global_a_offset = blockIdx.y * BM * K;
        unsigned int global_b_offset = blockIdx.x * BN;
        unsigned int global_c_offset = blockIdx.y * BM * N + blockIdx.x * BN;

        unsigned int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
        unsigned int local_x = thread_index % BN;
        unsigned int local_y = thread_index / BN;

        float sum = 0.0;
        for (unsigned int k = 0; k < K; k += BK) {
            shared_a[local_y * BK + local_x] = matrix_a[global_a_offset + local_y * K + k + local_x];
            shared_b[local_y * BN + local_x] = matrix_b[global_b_offset + (local_y + k) * N + local_x];

            __syncthreads();

            for (unsigned int i = 0; i < BK; i++) {
                sum += shared_a[local_y * BK + i] * shared_b[i * BN + local_x];
            }
            __syncthreads();
        }
        matrix_c[global_c_offset + local_y * N + local_x] =
                alpha * sum + beta * matrix_c[global_c_offset + local_y * N + local_x];
    }

    template<
            const unsigned int M,
            const unsigned int N,
            const unsigned int K,
            const unsigned int BM,
            const unsigned int BN,
            const unsigned int BK,
            const unsigned int TM
    >
    __global__ void block_tiling_1d(
            const float alpha,
            const float *matrix_a,
            const float *matrix_b,
            const float beta,
            float *matrix_c
    ) {
        __shared__ float shared_a[BM * BK];
        __shared__ float shared_b[BK * BN];

        unsigned int thread_index = threadIdx.y * blockDim.x + threadIdx.x;

        unsigned int thread_num = BM * BN / TM;

        unsigned int tx = thread_index % BN;
        unsigned int ty = thread_index / BN * TM;

        unsigned int a_global_offset = blockIdx.y * BM * K;
        unsigned int b_global_offset = blockIdx.x * BN;
        const unsigned int c_global_offset = blockIdx.y * BM * N + blockIdx.x * BN;

        unsigned int a_tile_row = thread_index / BK;
        unsigned int a_tile_col = thread_index % BK;
        unsigned int a_tile_stride = thread_num / BK;

        unsigned int b_tile_row = thread_index / BN;
        unsigned int b_tile_col = thread_index % BN;
        unsigned int b_tile_stride = thread_num / BN;

        // Must be explicitly set to 0.0
        float c_tile[TM] = {0.0};
        float b_cache = 0.0;
        for (unsigned int k = 0; k < K; k += BK) {
            for (unsigned int i = 0; i < BM; i += a_tile_stride) {
                shared_a[(a_tile_row + i) * BK + a_tile_col] = matrix_a[a_global_offset + (a_tile_row + i) * K +
                                                                        a_tile_col];
            }
            for (unsigned int i = 0; i < BK; i += b_tile_stride) {
                shared_b[(b_tile_row + i) * BN + b_tile_col] = matrix_b[b_global_offset + (b_tile_row + i) * N +
                                                                        b_tile_col];
            }

            __syncthreads();
            a_global_offset += BK;
            b_global_offset += BK * N;

            for (unsigned int i = 0; i < BK; i++) {
                b_cache = shared_b[tx + i * BN];
                for (unsigned int j = 0; j < TM; j++) {
                    c_tile[j] += shared_a[(ty + j) * BK + i] * b_cache;
                }
            }
            __syncthreads();
        }
        for (unsigned int j = 0; j < TM; j++) {
            matrix_c[c_global_offset + (ty + j) * N + tx] =
                    alpha * c_tile[j] + beta * matrix_c[c_global_offset + (ty + j) * N + tx];
        }
    }

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
//        unsigned int thread_index = threadIdx.y * blockDim.x + threadIdx.x;

        unsigned int thread_index = threadIdx.x;
        int block_row_thread = BN / TN;
        int block_col_thread = BM / TM;
        int thread_num = block_row_thread * block_col_thread; // 一个线程负责计算block中TM*TN个元素

        int tx = (thread_index % block_row_thread) * TN;
//        int tx = thread_index % BN; // This is wrong
        int ty = (thread_index / block_row_thread) * TM;

        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];

        // 移动到当前block
        unsigned int a_global_offset = blockIdx.y * BM * K;
        unsigned int b_global_offset = blockIdx.x * BN;
        const unsigned int c_global_offset = blockIdx.y * BM * N + blockIdx.x * BN;

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

//        float tmp[TM][TN] = {0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；
        float c_tile[TM * TN] = {0.0};
        for (int k = 0; k < K; k += BK) {
            for (int i = 0; i < BM; i += a_tile_stride) {
                As[(a_tile_row + i) * BK + a_tile_col] = A[a_global_offset + (a_tile_row + i) * K + a_tile_col];
            }
            for (int i = 0; i < BK; i += b_tile_stride) {
                Bs[(b_tile_row + i) * BN + b_tile_col] = B[b_global_offset + (b_tile_row + i) * N + b_tile_col];
            }
            __syncthreads();
            a_global_offset += BK;
            b_global_offset += BK * N;

            for (int kk = 0; kk < BK; kk++) {
                for (int i = 0; i < TM; i++) {
                    for (int j = 0; j < TN; j++)
                        c_tile[i * TN + j] += As[(ty + i) * BK + kk] * Bs[tx + j + kk * BN];
                }
            }
            __syncthreads();
        }
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < TN; j++)
                C[c_global_offset + (ty + i) * N + tx + j] =
                        alpha * c_tile[i * TN + j] + beta * C[c_global_offset + (ty + i) * N + tx + j];
        }
    }

#define OFFSET(row, col, ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

    template<
            const unsigned int BM,
            const unsigned int BN,
            const unsigned int BK,
            const unsigned int TM,
            const unsigned int TN
    >
    __global__ void vectorize_block_tiling_2d(
            int M, int N, int K,
            float alpha,
            float *A,
            float *B,
            float beta,
            float *C
    ) {
        int bx = blockIdx.x;
        int by = blockIdx.y;

        const int block_row_thread = BN / TN;
        const int block_col_thread = BM / TM;
        const int thread_num = block_row_thread * block_col_thread; // 一个线程负责计算block中TM*TN个元素

        // 当前线程对应thread tile的左上角元素在block中的位置
        int tx = (threadIdx.x % block_row_thread) * TN;
        int ty = (threadIdx.x / block_row_thread) * TM;

        __shared__ float As[BK * BM];
        __shared__ float Bs[BK * BN];

        const int ldg_a_num = BK * BM / thread_num / 4; // 每个线程搬运4个浮点数，完成搬运至As需要所有线程搬运ldg_a_num轮
        const int ldg_b_num = BK * BN / thread_num / 4; // 每个线程搬运4个浮点数，完成搬运至Bs需要所有线程搬运ldg_b_num轮

        int a_tile_row = threadIdx.x / (BK / 4); // 每行4个字节作为一个内存块，当前线程负责第a_tile_row行的第a_tile_col个内存块的搬运
        int a_tile_col = threadIdx.x % (BK / 4) * 4;
        int a_tile_stride = BM / ldg_a_num; // 一共BM行，搬运ldg_a_num轮，每论搬运a_tile_stride行

        int b_tile_row = threadIdx.x / (BN / 4); // 每行4个字节作为一个内存块，当前线程负责第b_tile_row行的第b_tile_col个内存块的搬运
        int b_tile_col = threadIdx.x % (BN / 4) * 4;
        int b_tile_stride = BK / ldg_b_num; // 一共BK行，搬运ldg_b_num轮，每论搬运b_tile_stride行

        float accum[TM][TN] = {0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；

        // 计算ldg_a_num的所有参数必须全部是const，否则不能用来申明数组大小
        float ldg_a_reg[4 * ldg_a_num] = {0.}; // 每个线程搬运ldg_a_num轮，寄存器缓存ldg_a_num个float4元素，用于转置As矩阵

        float a_frag[TM];  // 缓存As共享内存
        float b_frag[TN];  // 缓存Bs共享内存

        // 移动到当前block
        A = &A[by * BM * K];
        B = &B[bx * BN];
        C = &C[by * BM * N + bx * BN];

        for (int k = 0; k < K; k += BK) {
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = i / a_tile_stride * 4;  // 第ldg_index轮
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                        FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
                // As转置存，其中ldg_a_reg做中间缓存，目的是读取时可以按FLOAT4读取
                As[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
                As[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
                As[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
                As[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
            }
            for (int i = 0; i < BK; i += b_tile_stride) {
                FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                        FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]); // 不需要转置
            }
            __syncthreads();
            A += BK;
            B += BK * N;
            for (int i = 0; i < BK; i++) {
                for (int m = 0; m < TM; m += 4) {
                    FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(i, ty + m, BM)]); // 偏移到当前thread tile
                }
                for (int n = 0; n < TN; n += 4) {
                    FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(i, tx + n, BN)]); // 偏移到当前thread tile
                }
                for (int m = 0; m < TM; m++) {
                    for (int n = 0; n < TN; n++) {
                        accum[m][n] += a_frag[m] * b_frag[n];
                    }
                }
            }
            __syncthreads();
        }
        for (int m = 0; m < TM; m++) {
            for (int n = 0; n < TN; n += 4) {
                float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
                //float4 atmp = FETCH_FLOAT4(accum[m][n]);
                ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
                ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
                ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
                ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
                FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
            }
        }
    }
}

#endif //CUDA_KERNEL_MATMUL_CUH
