*this branch is the last working version. please run tests on this branch*
# CUDA Radix Sort

Histogram base 16 radix sort running on the GPU.
Benchmarked against std::sort and a single threaded naive approach as a CPU baseline.

## Files
- `radix_device.cu` — GPU kernels and device functions
- `radix_device.cuh` — header
- `radix_host.cpp` — host code, timing, and verification

## Requirements
- NVIDIA GPU, adjust makefile to match your GPU's architecture.

## Build & Run
make
./radix
