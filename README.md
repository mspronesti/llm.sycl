# llm.sycl
A cross-architecture porting of [Andrej Karphaty's llm.c](https://github.com/karpathy/llm.c) to SYCL/Intel OneAPI.

## Quick start

### Quick start (GPU, fp32 only)
Run the 1 GPU, fp32 code like this
```shell
chmod u+x ./download_starter_pack.sh
./download_starter_pack.sh
make train_gpt2_fp32
./train_gpt2_fp32
```

### Quick start (single kernels)

The `sycl` directory contains a number of standalone kernels that can be compiled and run independently. These are the building blocks of the full model and the `train_*` files.

Let's take `attention_forward.cpp` as an example. The following will compile the attention forward pass kernel for Intel's hardware:

```shell
cd sycl/
make attention_forward
```

Then run it with
```shell
./attention_forward
```

Use the `-DCUDA` and `-DCUDA_ARCH` flags to enable NVIDIA support. Similarly, `-DHIP` and `-DHIP_ARCH` for AMD support.





