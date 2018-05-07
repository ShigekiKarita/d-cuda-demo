.PHONY: all

CUDA_COMPUTE_CAPABILITY := 30
CUDA_BIT := $(shell getconf LONG_BIT)

all: saxpy.ptx

%.ptx: %.d
	ldc2 $< --mdcompute-targets=cuda-$(CUDA_COMPUTE_CAPABILITY)0 -mdcompute-file-prefix=$(shell basename -s .d $<) -H
	mv $(shell basename -s .d $<)_cuda$(CUDA_COMPUTE_CAPABILITY)0_$(CUDA_BIT).ptx $@
