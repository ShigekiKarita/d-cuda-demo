# Demo: Writing CUDA Kernel in D

Thanks to DCompute and LDC1.8.0-, we can write CUDA kernel in D.
This project aims to be minimal for writing CUDA kernels and calling CUDA Driver APIs.

## how to run

```d
$ curl -fsS https://dlang.org/install.sh | bash -s ldc-1.9.0
$ source ~/dlang/ldc-1.9.0/activate
$ make test
```

I have tested with

- LDC1.8.0/1.9.0 (prebuilt binary)
- CUDA9.1
- NVidia GTX760

## Links

- https://github.com/ldc-developers/ldc
- https://llvm.org/docs/NVPTXUsage.html
- https://llvm.org/docs/CompileCudaWithLLVM.html
