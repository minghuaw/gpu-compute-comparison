@group(0)
@binding(0)
var<storage, read> matrix_a: array<f32>;

@group(0)
@binding(1)
var<storage, read> matrix_b: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> matrix_c: array<f32>;

const M: u32 = 4096u;
const N: u32 = 4096u;
const K: u32 = 4096u;

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;

    if (row >= M) || (col >= N) {
        return;
    }

    var sum: f32 = 0.0f;
    for (var k = 0u; k < K; k += 1u) {
        let a_index = row * K + k;
        let b_index = k * N + col;
        sum += matrix_a[a_index] * matrix_b[b_index];
    }

    let c_index = row * N + col;
    matrix_c[c_index] = sum;
}