# Compiler
CC          := clang++
# Flags
CFLAGS      := -std=c++20 -fsycl -fno-finite-math-only
OPTIMIZE    := yes
CUDA        := no
CUDA_ARCH   := sm_70
HIP         := no
HIP_ARCH    := gfx908

# Optimization flag
ifeq ($(OPTIMIZE),yes)
    CFLAGS += -O3
endif
# Debug flag
ifeq ($(DEBUG),yes)
    CFLAGS += -g -DDEBUG
endif

# CUDA flag
ifeq ($(CUDA),yes)
    CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

# HIP flag
ifeq ($(HIP),yes)
    CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
endif

# Build rule for individual files
%: %.cpp
	$(CC) $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f $(wildcard *.o) $(wildcard main)

