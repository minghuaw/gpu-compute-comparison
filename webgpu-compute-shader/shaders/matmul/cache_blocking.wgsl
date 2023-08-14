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
const block_size_m: u32 = 8u;
const block_size_n: u32 = 8u;
const block_size_k: u32 = 8u;

var<workgroup> shared_a: array<f32, 64>;
var<workgroup> shared_b: array<f32, 64>;

@compute
@workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let global_a_offset = group_id.y * block_size_m * K;
    let global_b_offset = group_id.x * block_size_n;
    let global_c_offset = group_id.y * block_size_m * N + group_id.x * block_size_n;
    
    let local_x = local_index % block_size_n;
    let local_y = local_index / block_size_n;

    var sum = 0.0f;
    for (var k = 0u; k < K; k += block_size_k) {
        shared_a[local_y * block_size_k + local_x] = matrix_a[(global_a_offset + k) + local_y * K + local_x];
        shared_b[local_y * block_size_n + local_x] = matrix_b[(global_b_offset + k) + local_y * N + local_x];
        workgroupBarrier();
        for (var i = 0u; i < block_size_k; i++) {
            sum += shared_a[local_y * block_size_k + i] * shared_b[i * block_size_n + local_x];
        }
        workgroupBarrier();
    }
    matrix_c[global_c_offset + local_y * N + local_x] = sum;
}