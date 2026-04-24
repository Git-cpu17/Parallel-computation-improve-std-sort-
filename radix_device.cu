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

// base 16, so we have 4 bits per pass and 8 passed for 32-bit integers
#define BLOCK_SIZE 256
#define RADIX_BITS 4
#define RADIX_SIZE 16 // 2^4 = 16 buckets

__global__ void blockHist(int *arr, int *block_hist, int n, int shift)
{
    __shared__ int s_hist[RADIX_SIZE];
    int tx = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tx;

    if (tx < RADIX_SIZE)
        s_hist[tx] = 0;
    __syncthreads();

    if (gid < n)
    {
        int digit = (arr[gid] >> shift) & (RADIX_SIZE - 1);
        atomicAdd(&s_hist[digit], 1);
    }
    __syncthreads();

    // write to global
    if (tx < RADIX_SIZE)
    {
        block_hist[blockIdx.x * RADIX_SIZE + tx] = s_hist[tx];
    }
}

__global__ void scatter(int *arr, int *output, int *block_offsets, int n, int shift)
{
    int tx = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;

    // s_pos[d] = where this block's next element with digit d should go
    __shared__ int s_pos[RADIX_SIZE];

    // load this block's starting offset for each bucket from global memory
    // block_offsets is laid out bucket-major: [bucket][block]
    if (tx < RADIX_SIZE)
    {
        s_pos[tx] = block_offsets[tx * gridDim.x + blockIdx.x];
    }
    __syncthreads();

    // thread 0 of each block scatters its tile serially (preserves input order = stable)
    // blocks still run in parallel with each other
    if (tx == 0)
    {
        int tile_end = min(block_start + blockDim.x, n);
        for (int i = block_start; i < tile_end; i++)
        {
            int digit = (arr[i] >> shift) & (RADIX_SIZE - 1);
            output[s_pos[digit]++] = arr[i];
        }
    }
}

void cudaOptimizedRadixSort(int *d_arr, int n)
{
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *d_output, *d_block_hist, *d_block_offsets;
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_block_hist, blocks * RADIX_SIZE * sizeof(int));
    cudaMalloc(&d_block_offsets, blocks * RADIX_SIZE * sizeof(int));

    int *h_block_hist = (int *)malloc(blocks * RADIX_SIZE * sizeof(int));
    int *h_block_offsets = (int *)malloc(blocks * RADIX_SIZE * sizeof(int));

    for (int shift = 0; shift < 32; shift += RADIX_BITS)
    {
        // per-block histograms on GPU
        blockHist<<<blocks, BLOCK_SIZE>>>(d_arr, d_block_hist, n, shift);

        // pull histograms back to host for the prefix-sum step
        cudaMemcpy(h_block_hist, d_block_hist,
                   blocks * RADIX_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

        // build per-block starting offsets in BUCKET-MAJOR order:
        //   for each bucket d, block 0 starts at the running total,
        //   block 1 starts right after block 0's bucket-d elements, etc.
        // this way all bucket-0 elements come first, then bucket-1, etc.,
        // AND within bucket d, block 0's elements come before block 1's (stability across blocks)
        int running = 0;
        for (int d = 0; d < RADIX_SIZE; d++)
        {
            for (int b = 0; b < blocks; b++)
            {
                h_block_offsets[d * blocks + b] = running;
                running += h_block_hist[b * RADIX_SIZE + d];
            }
        }

        cudaMemcpy(d_block_offsets, h_block_offsets,
                   blocks * RADIX_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        // scatter: each block writes its tile into the right spots in output
        scatter<<<blocks, BLOCK_SIZE>>>(d_arr, d_output, d_block_offsets, n, shift);

        // swap: next pass reads from what we just wrote
        int *tmp = d_arr;
        d_arr = d_output;
        d_output = tmp;
    }

    cudaDeviceSynchronize();
    free(h_block_hist);
    free(h_block_offsets);
    cudaFree(d_output);
    cudaFree(d_block_hist);
    cudaFree(d_block_offsets);
    // note: d_arr and d_output got swapped around — freeing d_output frees
    // whichever buffer isn't the final result. After 8 passes (even), the
    // caller's original buffer holds the final sorted data.
}

// warmup kernel to initialize the GPU and avoid cold start overhead
__global__ void warmupKernel() {}

void cudaWarmup()
{
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}