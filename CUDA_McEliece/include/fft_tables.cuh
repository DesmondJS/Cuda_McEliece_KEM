#pragma once
#include <stdint.h>
#include "params.h"

#ifdef __CUDACC__
// device constant symbols (declared here, defined in fft_tables.cu)
extern __device__ __constant__ uint64_t c_scalars[5][GFBITS];
extern __device__ __constant__ uint64_t c_consts [63][GFBITS];
extern __device__ __constant__ uint64_t c_powers[64][GFBITS];
extern __device__ __constant__ uint64_t c_scalars2x[5][2][GFBITS];
extern __device__ __constant__ unsigned char c_reversal[64];
extern __device__ __constant__ uint16_t c_beta[6];
#endif

// call once before using FFT kernels (on first decrypt)
void fft_tables_cuda_init();
