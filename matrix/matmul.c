/*
 * Matrix multiplication: naive, OpenMP, pretransposed, pretransposed + OpenMP.
 * Based on the four algorithms described in CSC 447 Assignment 3 (LAU, 2024).
 * The report provided the benchmark main() and algorithm definitions, rather
 * than full source bodies, so these functions are reconstructed for this repo.
 */
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_int(const char *s, int *result) {
    char *end=NULL; errno=0; long n=strtol(s,&end,10);
    if (errno || end==s || *end || n<=0 || n>10000) return 0;
    *result=(int)n; return 1;
}
static void *xmalloc(size_t sz) {
    void *p=malloc(sz);
    if(!p) {fprintf(stderr,"Out of memory (%zu bytes).\n",sz);exit(1);}
    return p;
}
static void fill_inputs(double *a,double *b,size_t elements) {
    uint32_t seed=0x1234ABCDu;
    for(size_t i=0;i<elements;i++) {
        seed=1664525u*seed+1013904223u;
        a[i]=((seed>>8)&0xffffu)/65535.0;
        seed=1664525u*seed+1013904223u;
        b[i]=((seed>>8)&0xffffu)/65535.0;
    }
}
static void transpose_b(const double *b, double *bt,int n) {
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++) bt[(size_t)j*n+i]=b[(size_t)i*n+j];
}
static void multiply(const double *a,const double *b,double *c,int n,int parallel,int transposed) {
    /* In the transposed cases, b is the already-transposed B array. */
    if (parallel) {
        #pragma omp parallel for schedule(static)
        for(int i=0;i<n;i++) {
            for(int j=0;j<n;j++) {
                double sum=0.0;
                for(int k=0;k<n;k++)
                    sum+=a[(size_t)i*n+k] * (transposed?b[(size_t)j*n+k]:b[(size_t)k*n+j]);
                c[(size_t)i*n+j]=sum;
            }
        }
    } else {
        for(int i=0;i<n;i++) {
            for(int j=0;j<n;j++) {
                double sum=0.0;
                for(int k=0;k<n;k++)
                    sum+=a[(size_t)i*n+k] * (transposed?b[(size_t)j*n+k]:b[(size_t)k*n+j]);
                c[(size_t)i*n+j]=sum;
            }
        }
    }
}
static int verify(const double *got,const double *ref,size_t count,double *maxerror) {
    double max=0.0;
    for(size_t i=0;i<count;i++) {
        double err=fabs(got[i]-ref[i]);
        if(err>max)max=err;
        if(!isfinite(got[i])||err>1e-9 * (1.0+fabs(ref[i]))) {*maxerror=max;return 0;}
    }
    *maxerror=max;return 1;
}
int main(int argc,char **argv) {
    int n=256,threads=4,repeats=3;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"--help")) {
            puts("Usage: matmul [--size N] [--threads N] [--repeats N]");return 0;
        }
        if(i+1>=argc){fprintf(stderr,"Missing value: %s\n",argv[i]);return 2;}
        const char *arg=argv[i],*value=argv[++i];
        int *target=!strcmp(arg,"--size")?&n:!strcmp(arg,"--threads")?&threads:!strcmp(arg,"--repeats")?&repeats:NULL;
        if(!target||!parse_int(value,target)){fprintf(stderr,"Bad argument: %s %s\n",arg,value);return 2;}
    }
    if(n>2048 || threads>256 || repeats>50) {fprintf(stderr,"Request too large.\n");return 2;}
    size_t cells=(size_t)n*n,bytes=cells*sizeof(double);
    double *a=xmalloc(bytes),*b=xmalloc(bytes),*bt=xmalloc(bytes),*c=xmalloc(bytes),*ref=xmalloc(bytes);
    fill_inputs(a,b,cells);
    omp_set_dynamic(0);omp_set_num_threads(threads);
    multiply(a,b,ref,n,0,0);
    printf("variant,size,threads,trial,time_s,max_abs_error,correct\n");
    for(int variant=0;variant<4;++variant) {
        int parallel=(variant==1||variant==3),transposed=(variant==2||variant==3);
        const char *name=variant==0?"naive":variant==1?"openmp":variant==2?"transpose":"transpose-openmp";
        for(int r=1;r<=repeats;r++) {
            const double start=omp_get_wtime();
            if(transposed)transpose_b(b,bt,n);
            multiply(a,transposed?bt:b,c,n,parallel,transposed);
            const double seconds=omp_get_wtime()-start;
            double maxerror=0.0;
            const int correct=verify(c,ref,cells,&maxerror);
            printf("%s,%d,%d,%d,%.9f,%.3e,%d\n",name,n,parallel?threads:1,r,seconds,maxerror,correct);
            if(!correct){fprintf(stderr,"ERROR: result mismatch in %s\n",name);free(a);free(b);free(bt);free(c);free(ref);return 1;}
        }
    }
    free(a);free(b);free(bt);free(c);free(ref);
    return 0;
}
