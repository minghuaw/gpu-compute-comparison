//
// Created by micha on 2023-08-19.
//

#ifndef CUDA_KERNEL_CONFIG_H
#define CUDA_KERNEL_CONFIG_H
namespace config {
    const unsigned int M = 4096;
    const unsigned int N = 4096;
    const unsigned int K = 4096;

    const unsigned int BM = 64;
    const unsigned int BN = 64;
    const unsigned int BK = 8;
    const unsigned int BLOCKSIZE = 32;

    const unsigned int TM = 8;
    const unsigned int TN = 8;

    const float alpha = 1.0;
    const float beta = 1.0;
}
#endif //CUDA_KERNEL_CONFIG_H
