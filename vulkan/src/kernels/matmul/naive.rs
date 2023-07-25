vulkano_shaders::shader! {
    ty: "compute",
    src: r"
        #version 460

        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

        layout(set = 0, binding = 0) buffer readonly MatrixA {
            int matrix_a[];
        };

        layout(set = 0, binding = 1) buffer readonly MatrixB {
            int matrix_b[];
        };

        layout(set = 0, binding = 2) buffer writeonly MatrixC {
            int matrix_c[];
        };

        void main() {
            const uint row = gl_GlobalInvocationID.x;
            const uint col = gl_GlobalInvocationID.y;
            
            const uint M = 4096;
            const uint N = 4096;
            const uint K = 4096;

            if (row >= M || col >= N) {
                return;
            }

            int sum = 0;
            for (uint k = 0; k < K; k++) {
                const uint a_index = row * K + k;
                const uint b_index = k * N + col;
                sum += matrix_a[a_index] * matrix_b[b_index];
            }

            const uint c_index = row * N + col;
            matrix_c[c_index] = sum;
        }
    ",
}