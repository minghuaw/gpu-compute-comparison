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
const BM: u32 = 32u;
const BN: u32 = 32u;
const BK: u32 = 32u;

var<workgroup> shared_a: array<f32, 64>;
var<workgroup> shared_b: array<f32, 64>;

@compute
@workgroup_size(32, 32, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    var global_a_offset = group_id.y * BM * K;
    var global_b_offset = group_id.x * BN;
    let global_c_offset = group_id.y * BM * N + group_id.x * BN;
    
    let local_x = local_index % BN;
    let local_y = local_index / BN;

    var sum = 0.0f;
    for (var k = 0u; k < K; k += BK) {
        shared_a[local_y * BK + local_x] = matrix_a[global_a_offset + local_y * K + k + local_x];
        shared_b[local_y * BN + local_x] = matrix_b[global_b_offset + (local_y + k) * N + local_x];
        
        workgroupBarrier();

        for (var i = 0u; i < BK; i++) {
            sum += shared_a[local_y * BK + i] * shared_b[i * BN + local_x];
        }
        workgroupBarrier();
    }
    matrix_c[global_c_offset + local_y * N + local_x] = sum;
}