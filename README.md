# llm.sycl
A multi-platform porting of [Andrej Karphaty's CUDA kernels](https://github.com/karpathy/llm.c/tree/master/dev/cuda) to SYCL/Intel OneAPI, compatible with Intel, NVIDIA and AMD GPUs.

## Usage
Let's take `attention_forward.cpp` as an example. The following will compile the attention forward pass kernel for Intel's hardware:

```shell
make sycl/attention_forward
./sycl/attention_forward
```

use the `-DCUDA` and `-DCUDA_ARCH` flags to enable NVIDIA support. Similarly, `-DHIP` and `-DHIP_ARCH` for AMD support.





