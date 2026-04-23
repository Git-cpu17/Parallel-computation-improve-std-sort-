// CUDA implementation of Radix Sort https://www.geeksforgeeks.org/dsa/radix-sort/

#include <cuda_runtime.h>
#include "radix_device.cuh"
#include <stdio.h>

// unoptimized version
__device__ int getMax(int arr[], int n)
{
    int mx = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > mx)
            mx = arr[i];
    return mx;
}

__device__ void countSort(int arr[], int output[], int n, int exp)
{
    int i, count[10] = {0};

    for (i = 0; i < n; i++)
        count[(arr[i] / exp) % 10]++;

    for (i = 1; i < 10; i++)
        count[i] += count[i - 1];

    for (i = n - 1; i >= 0; i--)
    {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    for (i = 0; i < n; i++)
        arr[i] = output[i];
}

__device__ void radixsort(int arr[], int output[], int n)
{
    int m = getMax(arr, n);
    for (int exp = 1; m / exp > 0; exp *= 10)
        countSort(arr, output, n, exp);
}

// kernel
__global__ void radixSortKernel(int *arr, int *output, int n)
{
    radixsort(arr, output, n);
}

void cudaRadixSort(int *d_arr, int n)
{
    int *d_output;
    cudaMalloc(&d_output, n * sizeof(int));
    radixSortKernel<<<1, 1>>>(d_arr, d_output, n);
    cudaDeviceSynchronize();
    cudaFree(d_output);
}

// optimized version

// removed getMax, we know when to stop know cause of fixed # of passes

// base 16, so we have 4 bits per pass and 8 passed for 32-bit integers
__device__ void countSort_opt(int arr[], int output[], int n, int shift)
{
    int i, count[16] = {0};

    for (i = 0; i < n; i++)
        count[(arr[i] >> shift) & 15]++;

    for (i = 1; i < 16; i++)
        count[i] += count[i - 1];

    for (i = n - 1; i >= 0; i--)
    {
        output[count[(arr[i] >> shift) & 15] - 1] = arr[i];
        count[(arr[i] >> shift) & 15]--;
    }

    for (i = 0; i < n; i++)
        arr[i] = output[i];
}

__device__ void radixsort_opt(int arr[], int output[], int n)
{
    for (int shift = 0; shift < 32; shift += 4)
    {
        countSort_opt(arr, output, n, shift);
    }
}

__global__ void optimizedRadix(int *arr, int *output, int n)
{
    radixsort_opt(arr, output, n);
}

#define BLOCK_SIZE 256
#define RADIX_BITS 4
#define RADIX_SIZE 16   // 2^4 = 16 buckets

__global__ void optimizedRadixParallelHist(int *arr, int *output, int n, int shift)
{
    int tx = threadIdx.x;

    // s_hist is shared across all 256 threads in the block
    __shared__ int s_hist[RADIX_SIZE];

    // threads 0-15 each zero one bucket; rest are idle here
    if (tx < RADIX_SIZE)
        s_hist[tx] = 0;
    __syncthreads();

    for (int i = tx; i < n; i += blockDim.x)
        atomicAdd(&s_hist[(arr[i] >> shift) & (RADIX_SIZE - 1)], 1);
    __syncthreads();

    // same as countSort_opt
    if (tx == 0)
    {
        for (int i = 1; i < RADIX_SIZE; i++)
            s_hist[i] += s_hist[i - 1];

        for (int i = n - 1; i >= 0; i--)
        {
            int digit = (arr[i] >> shift) & (RADIX_SIZE - 1);
            output[--s_hist[digit]] = arr[i];
        }
        for (int i = 0; i < n; i++)
            arr[i] = output[i];
    }
}

void cudaOptimizedRadixSort(int *d_arr, int n)
{
    int *d_output;
    cudaMalloc(&d_output, n * sizeof(int));
    // one kernel launch per 4-bit pass (8 total) instead of one launch total
    for (int shift = 0; shift < 32; shift += RADIX_BITS)
    {
        optimizedRadixParallelHist<<<1, BLOCK_SIZE>>>(d_arr, d_output, n, shift);
        cudaDeviceSynchronize();
    }
    cudaFree(d_output);
}

// warmup kernel to initialize the GPU and avoid cold start overhead
__global__ void warmupKernel() {}

void cudaWarmup()
{
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}