@group(0)
@binding(0)
var<storage, read> a_global: array<f32>;

@group(0)
@binding(1)
var<storage, read> b_global: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> c_global: array<f32>;

const M: u32 = 4096u;
const N: u32 = 4096u;
const K: u32 = 4096u;

const BM: u32 = 32u;
const BN: u32 = 32u;
const BK: u32 = 32u;

const TM: u32 = 8u;

const alpha: f32 = 1.0;
const beta: f32 = 1.0;

var<workgroup> a_shared: array<f32, 1024>;
var<workgroup> b_shared: array<f32, 1024>;

// workgroup_size_x = BM / TM
// workgroup_size_y = BN
@compute
@workgroup_size(4, 32, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let thread_index = local_index;
    let thread_num = BM * BN / TM;

    let tx =  thread_index % BN;
    let ty = thread_index / BN * TM;

    var a_global_offset = group_id.y * BM * K;
    var b_global_offset = group_id.x * BN;
    let c_global_offset = group_id.y * BM * N + group_id.x * BN;

    let a_tile_row = thread_index / BK;
    let a_tile_col = thread_index % BK;
    let a_tile_stride = thread_num / BK;

    let b_tile_row = thread_index / BN;
    let b_tile_col = thread_index % BN;
    let b_tile_stride = thread_num / BN;

    // Explicitly initialize to zero. TODO: test if this is necessary
    var c_tile: array<f32, TM>;
    for (var i = 0u; i < TM; i++) {
        c_tile[i] = 0.0f;
    }
    var b_cache = 0.0f;

    for (var k = 0u; k < K; k += BK) {
        for (var i = 0u; i < BM; i += a_tile_stride) {
            a_shared[(a_tile_row + i) * BK + a_tile_col] = a_global[a_global_offset + (a_tile_row + i) * K + a_tile_col];
        }
        for (var i = 0u; i < BK; i += b_tile_stride) {
            b_shared[(b_tile_row + i) * BN + b_tile_col] = b_global[b_global_offset + (b_tile_row + i) * N + b_tile_col];
        }

        workgroupBarrier();
        a_global_offset += BK;
        b_global_offset += BK * N;

        for (var i = 0u; i < BK; i++) {
            b_cache = b_shared[tx + i * BN];
            for (var j = 0u; j < TM; j++) {
                c_tile[j] += a_shared[(ty + j) * BK + i] * b_cache;
            }
        }
        workgroupBarrier();
    }
    for (var j = 0u; j < TM; j++) {
        c_global[c_global_offset + (ty + j) * N + tx] = alpha * c_tile[j] + beta * c_global[c_global_offset + (ty + j) * N + tx];
    }
}