#include <iostream>

__global__ void hello_kernel(void) {}

int main() {
    hello_kernel<<<1, 1>>>();
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
