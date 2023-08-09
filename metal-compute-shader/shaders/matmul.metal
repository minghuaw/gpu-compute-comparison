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