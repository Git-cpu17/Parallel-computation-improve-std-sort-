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

__device__ void countSort_opt_shared(int arr[], int output[], int n, int shift)
{
    __shared__ int count[16];
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Initialize shared count
    if (tid < 16) count[tid] = 0;
    __syncthreads();

    // Each thread counts its portion
    for (int i = tid; i < n; i += blockSize)
        atomicAdd(&count[(arr[i] >> shift) & 15], 1);
    __syncthreads();

    // Prefix sum in shared memory
    if (tid == 0) {
        for (int i = 1; i < 16; i++)
            count[i] += count[i - 1];
    }
    __syncthreads();

    // Each thread places its elements
    for (int i = tid; i < n; i += blockSize)
    {
        int digit = (arr[i] >> shift) & 15;
        int pos = atomicSub(&count[digit], 1) - 1;
        output[pos] = arr[i];
    }
    __syncthreads();

    // Copy back
    for (int i = tid; i < n; i += blockSize)
        arr[i] = output[i];
}

__device__ void radixsort_opt_parallel(int arr[], int output[], int n)
{
    for (int shift = 0; shift < 32; shift += 4)
    {
        countSort_opt_shared(arr, output, n, shift);
    }
}

__global__ void optimizedRadixParallel(int *arr, int *output, int n)
{
    radixsort_opt_parallel(arr, output, n);
}

void cudaOptimizedRadixSort(int *d_arr, int n)
{
    int *d_output;
    cudaMalloc(&d_output, n * sizeof(int));
    // Use one block with 256 threads for small arrays
    int blockSize = min(256, n);
    optimizedRadixParallel<<<1, blockSize>>>(d_arr, d_output, n);
    cudaDeviceSynchronize();
    cudaFree(d_output);
}

// warmup kernel to initialize the GPU and avoid cold start overhead
__global__ void warmupKernel() {}

void cudaWarmup()
{
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}