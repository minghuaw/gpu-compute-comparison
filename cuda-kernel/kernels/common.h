//
// Created by micha on 2023-08-21.
//

#ifndef CUDA_KERNEL_COMMON_H
#define CUDA_KERNEL_COMMON_H

namespace common {
    const unsigned int M = 4096;
    const unsigned int N = 4096;
    const unsigned int K = 4096;

    // TODO: This can be passed in as a kernel argument
    // Keep these three the same for now. There is likely bug in indexing that will cause illegal access
    const unsigned int BM = 32;
    const unsigned int BN = 32;
    const unsigned int BK = 32;
    const unsigned int BLOCKSIZE = 32;

    const unsigned int TM = 8;
    const unsigned int TN = 8;

    const float alpha = 1.0;
    const float beta = 1.0;
}

#endif //CUDA_KERNEL_COMMON_H
