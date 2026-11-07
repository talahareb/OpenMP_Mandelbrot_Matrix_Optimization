/*
 * Mandelbrot rendering: sequential and OpenMP row-wise parallelization.
 * Coursework basis: CSC 447, LAU (2024). This version adds correct thread
 * scoping, a monotonic wall-clock timer and a reproducible command line.
 */
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *name) {
    fprintf(stderr,
      "Usage: %s [--mode serial|omp] [--threads N] [--schedule static|dynamic]\\n"
      "          [--chunk N] [--width N] [--height N] [--max-iter N]\\n"
      "          [--center-real X] [--center-imag Y] [--span X]\\n"
      "          [--output image.pgm] [--no-image]\\n", name);
}

static int parse_int(const char *s, int *result) {
    char *end = NULL;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (errno || end == s || *end || n <= 0 || n > 1000000) return 0;
    *result = (int)n;
    return 1;
}
static int parse_double(const char *s, double *result) {
    char *end = NULL;
    errno = 0;
    double x = strtod(s, &end);
    if (errno || end == s || *end || !isfinite(x)) return 0;
    *result = x;
    return 1;
}

static uint8_t mandelbrot_pixel(int col, int row, int w, int h, int maxiter,
                                 double cr0, double ci0, double span) {
    /* Follow coursework mapping: each axis spans [-2, 2] by default. */
    const double cr = cr0 + (col - w / 2.0) * span / w;
    const double ci = ci0 + (row - h / 2.0) * span / h;
    double zr = 0.0, zi = 0.0;
    int iter = 0;
    while (iter < maxiter && zr * zr + zi * zi < 4.0) {
        const double zr_new = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = zr_new;
        ++iter;
    }
    /* White inside, black far outside, matching the course's monochrome plots. */
    return (uint8_t)((iter >= maxiter) ? 255 : (254.0 * iter / maxiter));
}

int main(int argc, char **argv) {
    int width = 640, height = 480, maxiter = 255, threads = 4, chunk = 1;
    int parallel = 0, dynamic = 1, save_image = 1;
    double center_real = 0.0, center_imag = 0.0, span = 4.0;
    const char *output = "mandelbrot.pgm";
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--no-image")) {save_image = 0; continue;}
        if (!strcmp(arg, "--help")) {usage(argv[0]); return 0;}
        if (i + 1 >= argc) {usage(argv[0]); return 2;}
        const char *value = argv[++i];
        if (!strcmp(arg, "--mode")) {
            if (!strcmp(value, "omp")) parallel = 1;
            else if (!strcmp(value, "serial")) parallel = 0;
            else {usage(argv[0]); return 2;}
        } else if (!strcmp(arg, "--schedule")) {
            if (!strcmp(value, "dynamic")) dynamic = 1;
            else if (!strcmp(value, "static")) dynamic = 0;
            else {usage(argv[0]); return 2;}
        } else if (!strcmp(arg, "--width")) {if (!parse_int(value, &width)) return 2;}
        else if (!strcmp(arg, "--height")) {if (!parse_int(value, &height)) return 2;}
        else if (!strcmp(arg, "--max-iter")) {if (!parse_int(value, &maxiter)) return 2;}
        else if (!strcmp(arg, "--threads")) {if (!parse_int(value, &threads)) return 2;}
        else if (!strcmp(arg, "--chunk")) {if (!parse_int(value, &chunk)) return 2;}
        else if (!strcmp(arg, "--center-real")) {if (!parse_double(value, &center_real)) return 2;}
        else if (!strcmp(arg, "--center-imag")) {if (!parse_double(value, &center_imag)) return 2;}
        else if (!strcmp(arg, "--span")) {if (!parse_double(value, &span) || span <= 0) return 2;}
        else if (!strcmp(arg, "--output")) output = value;
        else {usage(argv[0]); return 2;}
    }
    if (width > 10000 || height > 10000 || maxiter > 100000 || threads > 256 ||
        (size_t)width > SIZE_MAX / (size_t)height) {fprintf(stderr,"Excessive dimensions.\n");return 2;}
    uint8_t *pixels = malloc((size_t)width * (size_t)height);
    if (!pixels) {perror("malloc");return 1;}
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    omp_set_schedule(dynamic ? omp_sched_dynamic : omp_sched_static, chunk);
    const double started = omp_get_wtime();
    if (parallel) {
        #pragma omp parallel for schedule(runtime)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x)
                pixels[(size_t)y * width + x] =
                    mandelbrot_pixel(x,y,width,height,maxiter,center_real,center_imag,span);
        }
    } else {
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                pixels[(size_t)y * width + x] =
                    mandelbrot_pixel(x,y,width,height,maxiter,center_real,center_imag,span);
    }
    const double seconds = omp_get_wtime() - started;
    /* FNV-1a hash makes serial/parallel bitwise equality easy to verify. */
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t k = 0, n = (size_t)width * height; k < n; ++k) {
        hash ^= pixels[k];
        hash *= UINT64_C(1099511628211);
    }
    if (save_image) {
        FILE *f = fopen(output, "wb");
        if (!f) {perror(output);free(pixels);return 1;}
        fprintf(f,"P5\n%d %d\n255\n",width,height);
        if (fwrite(pixels,1,(size_t)width * height,f) != (size_t)width * height) {
            fprintf(stderr,"Failed to write image.\n");fclose(f);free(pixels);return 1;
        }
        fclose(f);
    }
    printf("mode=%s schedule=%s threads=%d width=%d height=%d iter=%d time_s=%.8f hash=%016llx\n",
           parallel ? "omp":"serial",dynamic ? "dynamic":"static",parallel ? threads:1,
           width,height,maxiter,seconds,(unsigned long long)hash);
    free(pixels);
    return 0;
}
