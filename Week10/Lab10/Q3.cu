#include <stdio.h>
#include <cuda.h>

#define N 16  // Array size (should be a power of 2 for simplicity)
#define BLOCK_SIZE 16

__global__ void inclusive_scan(int *d_in, int *d_out, int n) {
    __shared__ int temp[N];  // Shared memory for scan operation

    int tid = threadIdx.x;
    if (tid < n)
        temp[tid] = d_in[tid];
    __syncthreads();

    // Up-sweep (reduce) phase
    for (int offset = 1; offset < n; offset *= 2) {
        if (tid >= offset)
            temp[tid] += temp[tid - offset];
        __syncthreads();
    }

    // Store the result back in global memory
    if (tid < n)
        d_out[tid] = temp[tid];
}

int main() {
    int h_in[N], h_out[N];
    int *d_in, *d_out;

    // Initialize input data
    for (int i = 0; i < N; i++)
        h_in[i] = i + 1;  // Example: {1, 2, 3, ..., N}

    // Allocate memory on device
    cudaMalloc((void **)&d_in, N * sizeof(int));
    cudaMalloc((void **)&d_out, N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    inclusive_scan<<<1, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Input:  ");
    for (int i = 0; i < N; i++)
        printf("%d ", h_in[i]);
    printf("\n");
    
    printf("Output: ");
    for (int i = 0; i < N; i++)
        printf("%d ", h_out[i]);
    printf("\n");
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}
