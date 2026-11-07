# Recorded benchmark results

- `mandelbrot_timings.csv`: wall-clock timings and matching pixel hashes for sequential and OpenMP rendering. Three trials per configuration, fixed 1300 × 900 image and 255 iterations.
- `matrix_timings.csv`: wall-clock timings and numerical correctness for four implementations, matrix sizes 128/256/384, threads 1/2/4, three trials per configuration.
- `benchmark_environment.txt`: compilation/runtime environment for the recorded run.

All times are observed timings from the runnable C programs, not estimates or historical screenshots. Wall-clock timings depend on hardware and system load. The matrix comparison plots use four threads for OpenMP bars. Transpose preparation time is included; validation time is excluded.

Run `python3 scripts/benchmark.py` from the repository root to regenerate measurements, then `python3 scripts/visualize.py` to update plots.
