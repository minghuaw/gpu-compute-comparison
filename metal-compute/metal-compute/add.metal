//
//  naive.metal
//  metal-compute
//
//  Created by Minghua Wu on 2023-08-04.
//

#include <metal_stdlib>
using namespace metal;

kernel void add(const device float *left [[ buffer(0) ]],
                const device float *right [[ buffer(1) ]],
                device float *out [[ buffer(2) ]],
                uint id [[ thread_position_in_grid ]]) {
//    out[id] = in[id].x + in[id].y;
    const uint TMP = 1;
    out[id] = left[id] + right[id];
}
