// CUDA implementation of Radix Sort https://www.geeksforgeeks.org/dsa/radix-sort/

#include <cuda_runtime.h>
#include "radix_device.cuh"
#include <stdio.h>

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
    int i, count[10] = { 0 };

    for (i = 0; i < n; i++)
        count[(arr[i] / exp) % 10]++;

    for (i = 1; i < 10; i++)
        count[i] += count[i - 1];

    for (i = n - 1; i >= 0; i--) {
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

__global__ void warmupKernel() {}

void cudaWarmup()
{
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}