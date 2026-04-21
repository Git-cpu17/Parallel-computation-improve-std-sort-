# CUDA Radix Sort

Single-threaded LSD radix sort running on the GPU (1 block, 1 thread).
Benchmarked against std::sort as a CPU baseline.

## Files
- `radix_device.cu` — GPU kernels and device functions
- `radix_device.cuh` — header
- `radix_host.cpp` — host code, timing, and verification

## Requirements
- NVIDIA GPU, adjust makefile to match your GPU's architecture.

## Build & Run
make
./radix