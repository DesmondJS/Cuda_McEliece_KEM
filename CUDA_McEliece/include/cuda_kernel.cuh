#ifndef CUDA_KERNEL_CUH
#define CUDA_KERNEL_CUH

#include <stdint.h>
#include <cuda_runtime.h>

#define BATCH 1
#define KATNUM 1

#ifdef __cplusplus
extern "C" {
#endif

// List wrapper function callable by .cpp file.
//int crypto_kem_enc(unsigned char *c, unsigned char *key, unsigned char *pk);
int crypto_kem_enc_new(unsigned char *c, unsigned char *key, const unsigned char *pk);

//Host function declaration for CUDA kernels
void gen_e_cuda(unsigned char *e);
void syndrome_cuda(unsigned char *s, const unsigned char *pk, unsigned char *e);
void encrypt(unsigned char *s, const unsigned char *pk, unsigned char *e);
static inline void store8(unsigned char *out, uint64_t in);

#ifdef __cplusplus
}
#endif

//Kernel functions declarations
__global__ void load_gf_kernel(unsigned char *src, uint16_t *nums);
__global__ void filter_valid_indices(uint16_t *nums, uint16_t *ind, int *count);
__global__ void check_duplicates(uint16_t *ind, int *has_duplicates);
__global__ void bitonic_sort_kernel(uint16_t* data, int num_elements);
__global__ void compute_val(uint16_t *d_ind, uint64_t *d_val);
__global__ void set_bits(uint16_t *d_ind, uint64_t *d_val, uint64_t *d_e_int);
__global__ void warmup_kernel();
__global__ void syndrome_kernel(unsigned char *s, const unsigned char *pk, const unsigned char *e);
__global__ void syndrome_kernel_vecstyle(
    uint8_t* __restrict__ s,          // length SYND_BYTES, already initialized with e[0:SYND_BYTES]
    const uint8_t* __restrict__ pk,   // PK_NROWS * PK_ROW_BYTES
    const uint8_t* __restrict__ e);   // full e (SYS_N/8 bytes); we’ll use e + SYND_BYTES


#endif //CUDA_KERNEL_CUH