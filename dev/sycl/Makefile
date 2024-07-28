# Compiler
CC          := clang++
# Flags
CFLAGS      := -std=c++20 -fsycl -fno-finite-math-only -qmkl=parallel
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

TARGETS = adamw attention_backward attention_forward classifier_fused crossentropy_forward crossentropy_softmax_backward encoder_backward encoder_forward gelu_backward gelu_forward layernorm_backward layernorm_forward matmul_backward matmul_backward_bias matmul_forward residual_forward softmax_forward trimat_forward fused_residual_forward global_norm

all: $(TARGETS)

matmul_forward: matmul_forward.cpp
	$(CC) $(CFLAGS) $< -ldnnl -o $@

trimat_forward: trimat_forward.cpp
	$(CC) $(CFLAGS) $< -ldnnl -o $@

# Build rule for individual files
%: %.cpp
	$(CC) $(CFLAGS) $< -o $@

# Run all targets
run_all: all
	@for target in $(TARGETS); do \
		echo "\n========================================"; \
		echo "Running $$target ..."; \
		echo "========================================\n"; \
		./$$target; \
	done

.PHONY: clean
clean:
	rm -f $(wildcard *.o) $(wildcard main)

