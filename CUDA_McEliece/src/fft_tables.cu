#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>   // fprintf, stderr
#include <stdlib.h>  // abort

#include "../include/fft_tables.cuh"
#include "../include/params.h"

// Import the host arrays defined in fft_tables_host.c
extern const uint64_t H_scalars[5][GFBITS];
extern const uint64_t H_consts [63][GFBITS];
extern const uint64_t H_powers[64][GFBITS];
extern const uint64_t H_scalars2x[5][2][GFBITS];

// Device constant symbols (defined here exactly once)
__constant__ uint64_t c_scalars[5][GFBITS];
__constant__ uint64_t c_consts [63][GFBITS];
__constant__ uint64_t c_powers[64][GFBITS];
__constant__ uint64_t c_scalars2x[5][2][GFBITS];

__constant__ unsigned char c_reversal[64] = {
  0,32,16,48, 8,40,24,56, 4,36,20,52,12,44,28,60,
  2,34,18,50,10,42,26,58, 6,38,22,54,14,46,30,62,
  1,33,17,49, 9,41,25,57, 5,37,21,53,13,45,29,61,
  3,35,19,51,11,43,27,59, 7,39,23,55,15,47,31,63
};

// add alongside your other __constant__ definitions
__constant__ uint16_t c_beta[6] = { 8, 1300, 3408, 1354, 2341, 1154 };

static inline void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); abort(); }
}

void fft_tables_cuda_init() {
    static bool done = false;
    if (done) return;
    cuda_check(cudaMemcpyToSymbol(c_scalars,  H_scalars,  sizeof(H_scalars)));
    cuda_check(cudaMemcpyToSymbol(c_consts,   H_consts,   sizeof(H_consts)));
    cuda_check(cudaMemcpyToSymbol(c_powers,   H_powers,   sizeof(H_powers)));
    cuda_check(cudaMemcpyToSymbol(c_scalars2x,H_scalars2x,sizeof(H_scalars2x)));
    done = true;
}



