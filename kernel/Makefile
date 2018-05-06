.PHONY: all

all: saxpy.ptx

%.ptx: %.d
	ldc2 $< -march=nvptx64 -betterC -c -of=- -H \
	| sed "s/.target generic/.target sm_30/g" \
	| sed "s/.func/.entry/g" > $@


# -Hf=$ -mdcompute-targets=
