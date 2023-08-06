//
//  matmul.metal
//  metal-compute
//
//  Created by Minghua Wu on 2023-08-04.
//

#include <metal_stdlib>
using namespace metal;

kernel void matmul_naive(const device float *matrixA [[ buffer(0) ]],
                         const device float *matrixB [[ buffer(1) ]],
                         device float *matrixC [[ buffer(2) ]],
                         uint2 gid [[ thread_position_in_grid ]]) {
    
}
