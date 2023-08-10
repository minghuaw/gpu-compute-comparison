#include <iostream>

__global__ void my_kernel(void) {
	
}

int main(void) {
	my_kernel<<<1,1>>>();
	printf("hello cuda\n");
	return 0;
}