#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>

#define THREADS_PER_BLOCK 256

__global__ void generateRS(char *S, char *RS, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len * (len + 1) / 2) return;
    
    int pos = 0;
    for (int i = 0; i < len; i++) {
        for (int j = 0; j <= i; j++) {
            if (pos == idx) {
                RS[idx] = S[j];
                return;
            }
            pos++;
        }
    }
}

int main() {
    char h_S[256], h_RS[256 * 256] = {0};
    
    printf("Enter string S: ");
    scanf("%s", h_S);
    
    int len = strlen(h_S);
    int RS_len = len * (len + 1) / 2;
    
    char *d_S, *d_RS;
    cudaMalloc((void**)&d_S, len + 1);
    cudaMalloc((void**)&d_RS, RS_len + 1);
    
    cudaMemcpy(d_S, h_S, len + 1, cudaMemcpyHostToDevice);
    
    int blocks = (RS_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    generateRS<<<blocks, THREADS_PER_BLOCK>>>(d_S, d_RS, len);
    
    cudaMemcpy(h_RS, d_RS, RS_len, cudaMemcpyDeviceToHost);
    h_RS[RS_len] = '\0';
    
    printf("Output string RS: %s\n", h_RS);
    
    cudaFree(d_S);
    cudaFree(d_RS);
    
    return 0;
}
