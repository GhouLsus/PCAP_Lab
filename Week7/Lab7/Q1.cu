#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>

#define THREADS_PER_BLOCK 256

__device__ void strCpy(char *dest, const char *src, int length) {
    for (int i = 0; i < length; i++) {
        dest[i] = src[i];
    }
    dest[length] = '\0';
}

__global__ void countWordOccurrences(char *sentence, char *word, int *count, int sentenceLen, int wordLen) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > sentenceLen - wordLen) return;
    
    bool match = true;
    for (int i = 0; i < wordLen; i++) {
        if (sentence[idx + i] != word[i]) {
            match = false;
            break;
        }
    }
    
    if (match && (idx == 0 || sentence[idx - 1] == ' ') && (sentence[idx + wordLen] == ' ' || sentence[idx + wordLen] == '\0')) {
        atomicAdd(count, 1);
    }
}

int main() {
    char h_sentence[256];
    char h_word[50];
    int h_count = 0;
    
    printf("Enter a sentence: ");
    fgets(h_sentence, sizeof(h_sentence), stdin);
    h_sentence[strcspn(h_sentence, "\n")] = 0; // Remove trailing newline
    
    printf("Enter a word to count: ");
    scanf("%s", h_word);
    
    int sentenceLen = strlen(h_sentence);
    int wordLen = strlen(h_word);
    
    char *d_sentence, *d_word;
    int *d_count;
    
    cudaMalloc((void**)&d_sentence, sentenceLen + 1);
    cudaMalloc((void**)&d_word, wordLen + 1);
    cudaMalloc((void**)&d_count, sizeof(int));
    
    cudaMemcpy(d_sentence, h_sentence, sentenceLen + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, h_word, wordLen + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);
    
    int blocks = (sentenceLen + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    countWordOccurrences<<<blocks, THREADS_PER_BLOCK>>>(d_sentence, d_word, d_count, sentenceLen, wordLen);
    
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("The word '%s' appears %d times in the given sentence.\n", h_word, h_count);
    
    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);
    
    return 0;
}
