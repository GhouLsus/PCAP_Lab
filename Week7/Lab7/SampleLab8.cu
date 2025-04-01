#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void transpose(int *a, int *t, int rows, int cols) {
    int n = threadIdx.x;  
    int m = blockIdx.x;   

    if (m < rows && n < cols) {  
        t[n * rows + m] = a[m * cols + n];
    }
}

int main(void) {
    int *a, *t, *d_a, *d_t;
    int m, n, i, j;

    printf("Enter the number of rows (m): ");
    scanf("%d", &m);
    printf("Enter the number of columns (n): ");
    scanf("%d", &n);

    int size = sizeof(int) * m * n;
    

    a = (int*)malloc(size);
    t = (int*)malloc(size);


    printf("Enter the input matrix:\n");
    for (i = 0; i < m * n; i++) {
        scanf("%d", &a[i]);
    }

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_t, size);


    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    transpose<<<m, n>>>(d_a, d_t, m, n);
    cudaDeviceSynchronize();  
    cudaMemcpy(t, d_t, size, cudaMemcpyDeviceToHost);

    printf("Transposed matrix:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%d\t", t[i * m + j]);
        }
        printf("\n");
    }
    free(a);
    free(t);
    cudaFree(d_a);
    cudaFree(d_t);

    return 0;
}
