#include <metal_stdlib>
using namespace metal;

kernel void naive(
    const device float *matrix_a [[ buffer(0) ]],
    const device float *matrix_b [[ buffer(1) ]],
    device float *matrix_c [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    // TODO: This can be passed in as a kernel argument
    const uint M = 4096;
    const uint N = 4096;
    const uint K = 4096;

    // Somehow metal performs better when row and col are not swapped
    uint row = gid.x;
    uint col = gid.y;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += matrix_a[row * K + k] * matrix_b[k * N + col];
    }

    matrix_c[row * N + col] = sum;
}

kernel void cache_blocking(
    const device float *matrix_a [[ buffer(0) ]],
    const device float *matrix_b [[ buffer(1) ]],
    device float *matrix_c [[ buffer(2) ]],
    threadgroup float *shared_a [[ threadgroup(0) ]],
    threadgroup float *shared_b [[ threadgroup(1) ]],
    uint2 group_size [[ threads_per_threadgroup ]],
    uint2 group_id [[ threadgroup_position_in_grid ]],
    uint thread_index [[ thread_index_in_threadgroup ]]
) {
    const uint M = 4096;
    const uint N = 4096;
    const uint K = 4096;

    // TODO: This can be passed in as a kernel argument
    const uint BM = 32;
    const uint BN = 32;
    const uint BK = 32;

    uint global_a_offset = group_id.y * BM * K;
    uint global_b_offset = group_id.x * BN;
    uint global_c_offset = group_id.y * BM * N + group_id.x * BN;

    uint local_x = thread_index % BN;
    uint local_y = thread_index / BN;

    float sum = 0.0f;
    for (uint k = 0; k < K; k += BK) {
        shared_a[local_y * BK + local_x] = matrix_a[global_a_offset + local_y * K + k + local_x];
        shared_b[local_y * BN + local_x] = matrix_b[global_b_offset + (local_y + k) * N + local_x];
        
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sum = 0.0f;
        for (uint i = 0; i < BK; i++) {
            sum += shared_a[local_y * BK + i] * shared_b[i * BN + local_x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    matrix_c[global_c_offset + local_y * N + local_x] = sum;
}