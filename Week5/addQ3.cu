#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

__global__ void oddEvenSortKernel(int *d_arr, int n, bool isOddPhase) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n - 1) return;
    if ((isOddPhase && idx % 2 == 1) || (!isOddPhase && idx % 2 == 0)) {
        int temp;
        if (d_arr[idx] > d_arr[idx + 1]) {
            temp = d_arr[idx];
            d_arr[idx] = d_arr[idx + 1];
            d_arr[idx + 1] = temp;
        }
    }
}

void oddEvenTranspositionSort(int *h_arr, int n) {
    int *d_arr;
    size_t size = n * sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (int phase = 0; phase < n; phase++) {
        bool isOddPhase = (phase % 2 == 0);
        oddEvenSortKernel<<<blocks, threadsPerBlock>>>(d_arr, n, isOddPhase);

        cudaDeviceSynchronize();
    }
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}

int main() {
    int n = 10;
    int h_arr[] = {12, 11, 13, 5, 6, 7, 9, 3, 8, 2};

    std::cout << "Unsorted array: ";
    for (int i = 0; i < n; i++) {
        std::cout << h_arr[i] << " ";
    }
    std::cout << std::endl;
    oddEvenTranspositionSort(h_arr, n);

    std::cout << "Sorted array: ";
    for (int i = 0; i < n; i++) {
        std::cout << h_arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
