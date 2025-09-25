#ifndef CUDA_DECRYPT_CUH
#define CUDA_DECRYPT_CUH

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Decrypt (returns 0 on success; 1 on failure).
// NOTE: here `sk` is expected to point at the start of the McEliece
// secret-material region (i.e., in full key layouts you typically pass sk+40).
int decrypt(unsigned char *e, const unsigned char *sk, const unsigned char *s);

// KEM decapsulation wrapper (derives 32-byte shared key into `key`).
// Always returns 0 (reference-compatible).
int crypto_kem_dec(unsigned char *key,
                   const unsigned char *c,
                   const unsigned char *sk);

#ifdef __cplusplus
} // extern "C"
#endif

// =====================================================================
// Kernel declarations (layouts for reference):
// - recv64: 64 lanes of 64-bit words             -> uint64_t[64]
// - *_64x:  64 x GFBITS bit-sliced rows          -> uint64_t[64*GFBITS]
// - out2x:  2 x GFBITS bit-sliced rows           -> uint64_t[2*GFBITS]
// - e: packed error bits ( (1<<GFBITS)/8 bytes )
// =====================================================================

__global__ void k_preprocess(uint64_t* __restrict__ recv64,
                             const unsigned char* __restrict__ s);

__global__ void k_benes(uint64_t* __restrict__ r64,
                        const unsigned char* __restrict__ bits,
                        int rev);

__global__ void k_fft(uint64_t* __restrict__ out64x,
                      uint64_t* __restrict__ inGFBITS);

__global__ void k_sq_rows(uint64_t* __restrict__ eval64x);

__global__ void k_build_inv(uint64_t* __restrict__ inv64x,
                            const uint64_t* __restrict__ eval64x);

__global__ void k_and_scale(uint64_t* __restrict__ out64x,
                            const uint64_t* __restrict__ inv64x,
                            const uint64_t* __restrict__ recv64);

__global__ void k_fft_tr(uint64_t* __restrict__ out2x,
                         uint64_t* __restrict__ in64x);

__global__ void k_form_error(uint64_t* __restrict__ error64,
                             const uint64_t* __restrict__ eval64x);

__global__ void k_postprocess(unsigned char* __restrict__ e,
                              const uint64_t* __restrict__ error64);

__global__ void k_weight_check(const uint64_t* __restrict__ Err64,
                               const unsigned char* __restrict__ E,
                               unsigned int* __restrict__ out_w0,
                               unsigned int* __restrict__ out_w1);

__global__ void k_synd_cmp(const uint64_t* __restrict__ s0_2x,
                           const uint64_t* __restrict__ s1_2x,
                           unsigned short* __restrict__ out_ok,
                           unsigned int* __restrict__ d_flag,   
                           unsigned int* __restrict__ d_done);

#endif // CUDA_DECRYPT_CUH
