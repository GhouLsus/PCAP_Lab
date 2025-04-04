#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void add(int *a, int *b, int *c) {
    c[0] = a[0] + b[0];
}

int main(void) {
    int a, b, c;
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    a = 3;
    b = 5;

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    add<<<1, 1>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    printf("Result: %d\n", c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
