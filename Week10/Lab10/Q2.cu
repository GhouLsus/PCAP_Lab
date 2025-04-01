#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256   
#define FILTER_WIDTH 5 

__constant__ float d_filter[FILTER_WIDTH];

__global__ void convolution1D(float *d_input, float *d_output, int inputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    float result = 0.0f;

    if (idx < inputSize) {
        for (int i = 0; i < FILTER_WIDTH; i++) {
            int inputIdx = idx + i - FILTER_WIDTH / 2; 
            if (inputIdx >= 0 && inputIdx < inputSize) {
                result += d_input[inputIdx] * d_filter[i];
            }
        }
        d_output[idx] = result;
    }
}

int main() {
    int inputSize;
    printf("Enter the size of the input array: ");
    scanf("%d", &inputSize);

    float *h_input = (float *)malloc(inputSize * sizeof(float));
    float *h_output = (float *)malloc(inputSize * sizeof(float));
    float h_filter[FILTER_WIDTH] = {0.2, 0.4, 0.6, 0.4, 0.2}; 

    printf("Enter the elements of the input array:\n");
    for (int i = 0; i < inputSize; i++) {
        scanf("%f", &h_input[i]);
    }

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, inputSize * sizeof(float));
    cudaMalloc((void **)&d_output, inputSize * sizeof(float));

    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_filter, h_filter, FILTER_WIDTH * sizeof(float));

    int numBlocks = (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    convolution1D<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, inputSize);

    cudaMemcpy(h_output, d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result of 1D convolution:\n");
    for (int i = 0; i < inputSize; i++) {
        printf("%.2f ", h_output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
