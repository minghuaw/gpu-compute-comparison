#version 450
#pragma shader_stage(compute)

layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer readonly MatrixA {
    int[1<<20] matrix_a;
};

layout(set = 0, binding = 1) buffer readonly MatrixB {
    int[1<<20] matrix_b;
};

layout(set = 0, binding = 2) buffer writeonly MatrixC {
    int[1<<20] matrix_c;
};

void main() {
    const uint row = gl_GlobalInvocationID.x;
    const uint col = gl_GlobalInvocationID.y;

    if(row >= 4096 || col >= 4096) {
        return;
    }

    int sum = 0;
    for(uint i = 0; i < 4096; i++) {
        sum += matrix_a[row * 4096 + i] * matrix_b[i * 4096 + col];
    }

    matrix_c[row * 4096 + col] = sum;
}
