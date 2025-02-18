#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 5  // Number of binary numbers to process

__global__ void onesComplement(int *input, int *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        output[tid] = ~input[tid];  // Bitwise NOT operation
    }
}

// Function to print binary representation of an integer
void printBinary(int num) {
    for (int i = 31; i >= 0; i--) {
        printf("%d", (num >> i) & 1);
    }
}

int main() {
    int *h_input, *h_output;
    int *d_input, *d_output;
    
    // Allocate host memory
    h_input = (int*)malloc(N * sizeof(int));
    h_output = (int*)malloc(N * sizeof(int));
    
    // Initialize input with random binary numbers
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() & 0xFFFF;  // Generate 16-bit random numbers
    }
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    onesComplement<<<gridSize, blockSize>>>(d_input, d_output, N);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Input and One's Complement:\n");
    for (int i = 0; i < N; i++) {
        printf("Input:      ");
        printBinary(h_input[i]);
        printf("\nComplement: ");
        printBinary(h_output[i]);
        printf("\n\n");
    }
    
    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
