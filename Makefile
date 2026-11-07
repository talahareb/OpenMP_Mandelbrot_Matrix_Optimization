CC ?= gcc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -fopenmp
LDLIBS ?= -lm
.PHONY: all clean test
all: mandelbrot/mandelbrot matrix/matmul
mandelbrot/mandelbrot: mandelbrot/mandelbrot.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)
matrix/matmul: matrix/matmul.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)
test: all
	python3 scripts/test_correctness.py
clean:
	rm -f mandelbrot/mandelbrot matrix/matmul
