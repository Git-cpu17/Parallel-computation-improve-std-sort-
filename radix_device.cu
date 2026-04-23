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

void cudaOptimizedRadixSort(int *d_arr, int n)
{
    int *d_output;
    cudaMalloc(&d_output, n * sizeof(int));
    optimizedRadix<<<1, 1>>>(d_arr, d_output, n);
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