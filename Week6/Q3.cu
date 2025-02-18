#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void oddEvenSort(int *arr, int n, int *isSorted) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int phase;

    for (phase = 0; phase < n; phase++) {
        if (tid % 2 == phase % 2 && tid < n - 1) {
            if (arr[tid] > arr[tid + 1]) {
                int temp = arr[tid];
                arr[tid] = arr[tid + 1];
                arr[tid + 1] = temp;
                *isSorted = 0;
            }
        }
        __syncthreads();
    }
}

void printArray(int *arr, int n, const char *message) {
    printf("%s", message);
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    const int n = 20; // Reduced size for better visibility in output
    int h_arr[n];
    int *d_arr, *d_isSorted;
    int h_isSorted;

    // Seed the random number generator
    srand(time(NULL));

    // Initialize array with random values
    for (int i = 0; i < n; i++) {
        h_arr[i] = rand() % 100; // Values between 0 and 99
    }

    // Print the input array
    printArray(h_arr, n, "Input array:  ");

    // Allocate device memory
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_isSorted, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    do {
        h_isSorted = 1;
        cudaMemcpy(d_isSorted, &h_isSorted, sizeof(int), cudaMemcpyHostToDevice);

        oddEvenSort<<<gridSize, blockSize>>>(d_arr, n, d_isSorted);

        cudaMemcpy(&h_isSorted, d_isSorted, sizeof(int), cudaMemcpyDeviceToHost);
    } while (!h_isSorted);

    // Copy result back to host
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted array
    printArray(h_arr, n, "Sorted array: ");

    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_isSorted);

    return 0;
}
