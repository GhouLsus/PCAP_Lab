#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void parallelSelectionSort(int *arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }
        
        if (tid == i) {
            int temp = arr[i];
            arr[i] = arr[minIdx];
            arr[minIdx] = temp;
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
    int *h_arr = (int*)malloc(n * sizeof(int));
    int *d_arr;
    
    // Seed the random number generator
    srand(time(NULL));
    
    // Initialize array with random values
    for (int i = 0; i < n; i++) {
        h_arr[i] = rand() % 100; // Values between 0 and 99
    }
    
    // Print the input array
    printArray(h_arr, n, "Input array:  ");
    
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    parallelSelectionSort<<<gridSize, blockSize>>>(d_arr, n);
    
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print the sorted array
    printArray(h_arr, n, "Sorted array: ");
    
    cudaFree(d_arr);
    free(h_arr);
    
    return 0;
}
