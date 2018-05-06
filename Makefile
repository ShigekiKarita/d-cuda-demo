.PHONY: test clean kernel

test: kernel
	dub run

kernel:
	$(MAKE) -C kernel

clean:
	rm -rfv **/*.di **/*.ptx

include kernel/Makefile
