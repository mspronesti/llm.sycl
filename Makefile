CC          := icpx
CFLAGS      := -std=c++20 -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
SYCL_FLAGS  := -fsycl -qmkl=parallel
LDFLAGS     := -ldnnl
CFLAGS_COND := -march=native

# Debug flag
ifeq ($(DEBUG),yes)
    CFLAGS += -g -DDEBUG
endif

#PRECISION ?= BF16
#VALID_PRECISIONS := FP32 FP16 BF16
#ifeq ($(filter $(PRECISION),$(VALID_PRECISIONS)),)
#  $(error Invalid precision $(PRECISION), valid precisions are $(VALID_PRECISIONS))
#endif
#ifeq ($(PRECISION), FP32)
#  PFLAGS = -DENABLE_FP32
#else ifeq ($(PRECISION), FP16)
#  PFLAGS = -DENABLE_FP16
#else
#  PFLAGS = -DENABLE_BF16
#endif

.PHONY: all train_gpt2_fp32 test_gpt2_fp32 # train_gpt2 test_gpt2

# Add targets
TARGETS = train_gpt2_fp32 test_gpt2_fp32 # train_gpt2 test_gpt2

$(info ---------------------------------------------)

all: $(TARGETS)

train_gpt2_fp32: train_gpt2_fp32.cpp
	$(CC) -O3  $(SYCL_FLAGS) $< $(LDFLAGS) -o $@

test_gpt2_fp32: test_gpt2_fp32.cpp
	$(CC) -O3  $(SYCL_FLAGS) $< $(LDFLAGS) -o $@
