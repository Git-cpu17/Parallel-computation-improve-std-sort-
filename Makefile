CC = /usr/bin/g++

LD_FLAGS = -lrt

# change to your local directory
CUDA_PATH       ?= /usr/local/cuda-12.6
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

# CUDA code generation flags
# Update sm_86 to match your GPU's compute capability:
#   sm_75 = Turing     (RTX 20xx)
#   sm_86 = Ampere     (RTX 30xx)
#   sm_89 = Ada        (RTX 40xx)
#   sm_120 = Blackwell (RTX 50xx)
#   for the server, sm_86
GENCODE_FLAGS   := -gencode arch=compute_86,code=sm_86

# Common binaries
NVCC            ?= /usr/local/cuda-12.6/bin/nvcc # change to your local directory

# OS-specific build flags
ifeq ($(shell uname),Darwin)
	LDFLAGS     := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
	CCFLAGS     := -arch $(OS_ARCH)
else
	ifeq ($(OS_SIZE),32)
		LDFLAGS := -L$(CUDA_LIB_PATH) -lcudart
		CCFLAGS := -m32
	else
		CUDA_LIB_PATH := $(CUDA_LIB_PATH)64
		LDFLAGS       := -L$(CUDA_LIB_PATH) -lcudart
		CCFLAGS       := -m64
	endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
	NVCCFLAGS := -m32
else
	NVCCFLAGS := -m64
endif

TARGETS = radix

all: $(TARGETS)

radix: radix_host.cpp radix.o
	$(CC) $^ -o $@ -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)

radix.o: radix_device.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
