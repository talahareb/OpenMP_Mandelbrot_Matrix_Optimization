#!/usr/bin/env python3
"""Functional tests: serial-vs-parallel pixels and all four matrix variants."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def call(args):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=True).stdout


def main():
    with tempfile.TemporaryDirectory() as d:
        serial = Path(d) / "serial.pgm"
        parallel = Path(d) / "parallel.pgm"
        base = ["--width", "180", "--height", "120", "--max-iter", "250"]
        a = call(["./mandelbrot/mandelbrot", "--mode", "serial", *base, "--output", str(serial)])
        for threads in (1, 2, 4, 8):
            for schedule in ("static", "dynamic"):
                b = call(["./mandelbrot/mandelbrot", "--mode", "omp", "--threads", str(threads),
                          "--schedule", schedule, *base, "--output", str(parallel)])
                assert serial.read_bytes() == parallel.read_bytes(), f"Mandelbrot mismatch: {threads} {schedule}"
        print("PASS: identical Mandelbrot pixels for sequential, 1/2/4/8 OpenMP threads, static/dynamic")
        for size in (31, 64, 97):
            for threads in (1, 2, 4):
                output = call(["./matrix/matmul", "--size", str(size), "--threads", str(threads), "--repeats", "2"])
                assert all(line.split(",")[-1] == "1" for line in output.strip().splitlines()[1:])
        print("PASS: all four matrix variants agree for odd/even sizes and 1/2/4 threads")
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    main()
