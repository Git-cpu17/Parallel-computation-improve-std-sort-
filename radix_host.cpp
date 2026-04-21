#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "radix_device.cuh"
#include <algorithm>

#define gpuErrChk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void checkSorted(const int *arr, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        if (arr[i] > arr[i + 1])
        {
            fprintf(stderr, "Sort failed at index %d: %d > %d\n",
                    i, arr[i], arr[i + 1]);
            exit(EXIT_FAILURE);
        }
    }
}

void randomFill(int *arr, int n)
{
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 100000;
}

void printArray(const int *arr, int n)
{
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main()
{
    srand(2016);

    cudaWarmup();

    // Sizes to benchmark, powers of 2
    const int sizes[] = {1024, 4096, 16384};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    cudaEvent_t start, stop;

#define START_TIMER()                       \
    {                                       \
        gpuErrChk(cudaEventCreate(&start)); \
        gpuErrChk(cudaEventCreate(&stop));  \
        gpuErrChk(cudaEventRecord(start));  \
    }

#define STOP_RECORD_TIMER(name)                              \
    {                                                        \
        gpuErrChk(cudaEventRecord(stop));                    \
        gpuErrChk(cudaEventSynchronize(stop));               \
        gpuErrChk(cudaEventElapsedTime(&name, start, stop)); \
        gpuErrChk(cudaEventDestroy(start));                  \
        gpuErrChk(cudaEventDestroy(stop));                   \
    }

    for (int s = 0; s < num_sizes; s++)
    {
        int n = sizes[s];
        float gpu_ms = -1;
        float cpu_ms = -1;

        // Allocate and fill host array
        int *h_arr = new int[n];
        randomFill(h_arr, n);

        // Allocate device memory and copy input
        int *d_arr;
        gpuErrChk(cudaMalloc(&d_arr, n * sizeof(int)));
        gpuErrChk(cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice));

        // Time the sort
        START_TIMER();
        cudaRadixSort(d_arr, n);
        gpuErrChk(cudaDeviceSynchronize());
        STOP_RECORD_TIMER(gpu_ms);

        // Copy result back and verify
        gpuErrChk(cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));
        checkSorted(h_arr, n);

        // std sort for coimparison
        randomFill(h_arr, n);
        START_TIMER();
        std::sort(h_arr, h_arr + n);
        STOP_RECORD_TIMER(cpu_ms);
        checkSorted(h_arr, n);
        printf("Size %6d | our radix:  %8.4f ms\n", n, gpu_ms);
        printf("Size %6d | std::sort:  %8.4f ms\n\n", n, cpu_ms);

        // Cleanup
        delete[] h_arr;
        gpuErrChk(cudaFree(d_arr));
    }

    return 0;
}
