#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void selectionSortRow(int *matrix, int numRows, int numCols) {
    int row = blockIdx.x;  
    int startIdx = row * numCols;

    for (int i = 0; i < numCols - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < numCols; j++) {
            if (matrix[startIdx + j] < matrix[startIdx + minIdx]) {
                minIdx = j;
            }
        }

        if (minIdx != i) {
            int temp = matrix[startIdx + i];
            matrix[startIdx + i] = matrix[startIdx + minIdx];
            matrix[startIdx + minIdx] = temp;
        }
    }
}

int main() {
    int numRows = 3;
    int numCols = 4;  
    int size = numRows * numCols * sizeof(int); 
    int h_matrix[12] = {
        12, 11, 13, 5,
        6, 9, 8, 7,
        3, 4, 1, 2
    };
    int *d_matrix;
    cudaMalloc((void**)&d_matrix, size);
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);
    selectionSortRow<<<numRows, 1>>>(d_matrix, numRows, numCols);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    std::cout << "Sorted Matrix (each row sorted):\n";
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            std::cout << h_matrix[i * numCols + j] << " ";
        }
        std::cout << std::endl;
    }
    cudaFree(d_matrix);

    return 0;
}
