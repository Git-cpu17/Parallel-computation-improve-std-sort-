#ifndef CUDA_RADIX_CUH
#define CUDA_RADIX_CUH

void cudaRadixSort(int *d_arr, int n);
void cudaOptimizedRadixSort(int *d_arr, int n);

void cudaWarmup();

#endif