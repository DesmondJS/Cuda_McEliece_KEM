#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <ctime>
#include <sstream>

#include "../include/params.h"
#include "../include/util.h"         // irr_load, load8/store8 (host)
#include "../include/gf.h"           // gf ops for BM
#include "../include/bm.h"           // CPU BM (sequential)
#include "../include/fft_tables.cuh" // Load the fft tables
#include "../include/crypto_hash.h"  //For the SHAKE256
#include "../include/cuda_decrypt.cuh"
#include "../include/fft.h"
#include "../include/fft_tr.h"
#include "../include/vec.h" 

// ================== Device Functions =======================


// XOR of two bit-sliced vectors. Used throughout FFT/BENES steps
__device__ __forceinline__ void d_vec_add(uint64_t* __restrict__ z,
                                          const uint64_t* __restrict__ x,
                                          const uint64_t* __restrict__ y) {
    #pragma unroll
    for (int b = 0; b < GFBITS; b++) z[b] = x[b] ^ y[b];
}

// 64x64 bit-matrix transpose (device) — same as transpose.h
// Reorders 64 lanes for butterflies/bit-reversal; used in FFT_tr and Benes.
__device__ __forceinline__ void d_transpose_64x64(uint64_t* out, const uint64_t* in) {
    static const uint64_t masks[6][2] = {
        {0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL},
        {0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL},
        {0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL},
        {0x00FF00FF00FF00FFULL, 0xFF00FF00FF00FF00ULL},
        {0x0000FFFF0000FFFFULL, 0xFFFF0000FFFF0000ULL},
        {0x00000000FFFFFFFFULL, 0xFFFFFFFF00000000ULL}
    };

    uint64_t tmp[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) tmp[i] = in[i];

    for (int d = 5; d >= 0; d--) {
        int s = 1 << d;
        for (int i = 0; i < 64; i += s*2) {
            for (int j = i; j < i + s; j++) {
                uint64_t x = (tmp[j]     & masks[d][0]) | ((tmp[j+s] & masks[d][0]) << s);
                uint64_t y = ((tmp[j]    & masks[d][1]) >> s)        | (tmp[j+s] & masks[d][1]);
                tmp[j]   = x;
                tmp[j+s] = y;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 64; i++) out[i] = tmp[i];
}

// popcount for 64-bit
// Utility used for weight checks / quick bit counts.
__device__ __forceinline__ int d_popcount64(uint64_t x){ return __popcll(x); }

// ===== Device bitsliced field ops =====
using vec = uint64_t;


// d_vec_copy: Copy a GFBITS-length bit-sliced vector (vec = uint64_t).
// Simple unrolled copy; used in GF ladder routines.
__device__ __forceinline__ void d_vec_copy(vec* __restrict__ out,
                                           const vec* __restrict__ in) {
    #pragma unroll
    for (int i = 0; i < GFBITS; i++) out[i] = in[i];
}

/* bitsliced multiplication in GF(2^12) with modulus x^12 + x^3 + 1 */
// Computes h = f * g under modulus (x^12 + x^3 + 1). Core for FFT steps.
__device__ __forceinline__ void d_vec_mul(vec* __restrict__ h,
                                          const vec* __restrict__ f,
                                          const vec* __restrict__ g) {
    vec buf[2*GFBITS - 1];

    #pragma unroll
    for (int i = 0; i < 2*GFBITS - 1; i++) buf[i] = 0;

    #pragma unroll
    for (int i = 0; i < GFBITS; i++) {
        #pragma unroll
        for (int j = 0; j < GFBITS; j++) {
            buf[i + j] ^= (f[i] & g[j]);
        }
    }

    // fold high terms per x^12 = x^3 + 1
    for (int i = 2*GFBITS - 2; i >= GFBITS; i--) {
        buf[i - GFBITS + 3] ^= buf[i];
        buf[i - GFBITS + 0] ^= buf[i];
    }

    #pragma unroll
    for (int i = 0; i < GFBITS; i++) h[i] = buf[i];
}

/* bitsliced squaring: same wiring as vec_sq() */
// Fixed wiring for squaring in GF(2^m).
__device__ __forceinline__ void d_vec_sq(vec* __restrict__ out,
                                         const vec* __restrict__ in) {
    vec r[GFBITS];
    r[0]  = in[0] ^ in[6];
    r[1]  = in[11];
    r[2]  = in[1] ^ in[7];
    r[3]  = in[6];
    r[4]  = in[2] ^ in[11] ^ in[8];
    r[5]  = in[7];
    r[6]  = in[3] ^ in[9];
    r[7]  = in[8];
    r[8]  = in[4] ^ in[10];
    r[9]  = in[9];
    r[10] = in[5] ^ in[11];
    r[11] = in[10];

    #pragma unroll
    for (int i = 0; i < GFBITS; i++) out[i] = r[i];
}

/* bitsliced inversion via fixed ladder (same as vec_inv()) */
// Produces out = in^{-1}; used when building per-coordinate inverses.
__device__ __forceinline__ void d_vec_inv(vec* __restrict__ out,
                                          const vec* __restrict__ in) {
    vec tmp_11[GFBITS];
    vec tmp_1111[GFBITS];

    d_vec_copy(out, in);

    d_vec_sq(out, out);
    d_vec_mul(tmp_11, out, in);           // 11

    d_vec_sq(out, tmp_11);
    d_vec_sq(out, out);
    d_vec_mul(tmp_1111, out, tmp_11);     // 1111

    d_vec_sq(out, tmp_1111);
    d_vec_sq(out, out);
    d_vec_sq(out, out);
    d_vec_sq(out, out);
    d_vec_mul(out, out, tmp_1111);        // 11111111

    d_vec_sq(out, out);
    d_vec_sq(out, out);
    d_vec_mul(out, out, tmp_11);          // 1111111111

    d_vec_sq(out, out);
    d_vec_mul(out, out, in);              // 11111111111

    d_vec_sq(out, out);                   // 111111111110
}

// d_vec_setbits: Broadcast 1-bit 'b' to an all-ones/all-zeros mask.
// Returns ~0 if b=1 else 0; handy for masking in bit-sliced logic.
__device__ __forceinline__ vec d_vec_setbits(uint64_t b) { return ~uint64_t(0) * (b & 1ULL); }

// d_vec_or_reduce: OR-reduction across GFBITS slices.
// Returns OR_j a[j]; used to test "is evaluation == 0?" in error formation.
__device__ __forceinline__ vec d_vec_or_reduce(const vec* __restrict__ a) {
    vec r = a[0];
    #pragma unroll
    for (int i = 1; i < GFBITS; i++) r |= a[i];
    return r;
}

// d_load4: Little-endian unaligned 32-bit load from bytes.
// Used to read Benes conditions packed as bytes from secret key.
__device__ __forceinline__ uint32_t d_load4(const unsigned char* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}

// d_load8: Little-endian unaligned 64-bit load from bytes.
// Used in Benes/FFT data loads where alignment is not guaranteed.
__device__ __forceinline__ uint64_t d_load8(const unsigned char* p) {
    uint64_t v=0;
    #pragma unroll
    for (int i=7;i>=0;i--) { v<<=8; v|=p[i]; }
    return v;
}

// Parallel version of layer(): 32 disjoint pairs per layer
// d_layer_parallel: One Benes "layer" (butterfly) in parallel.
// Applies conditional swaps for 32 disjoint (j, j+s) pairs at stage 'lgs'.
// 'bits' holds masks for the layer; threads split the 32 pairs.
__device__ __forceinline__ void d_layer_parallel(uint64_t* __restrict__ data, const uint64_t* __restrict__ bits, 
                                                 int lgs, int t, int nt)
{
    const int s     = 1 << lgs;
    const int pairs = 32;                // 64 lanes form 32 (j, j+s) pairs
    for (int idx = t; idx < pairs; idx += nt) {
        const int blk = idx / s;
        const int off = idx % s;
        const int j   = blk * (2*s) + off;

        uint64_t d = (data[j] ^ data[j + s]) & bits[idx];
        data[j]   ^= d;
        data[j+s] ^= d;
    }
}

// d_broadcast_and_beta: CPU-accurate broadcast+beta accumulation for FFT_tr.
// Reconstructs out01[0], out01[1] from 64xGFBITS matrix using c_beta masks.
// Matches the CPU reference order exactly (important for correctness).
static __device__ __forceinline__
void d_broadcast_and_beta(uint64_t out01[2][GFBITS],
                          uint64_t in[64][GFBITS])
{
    uint64_t pre[6][GFBITS];

    // pre / accumulation chain (identical to CPU order)
    d_vec_copy(pre[0], in[32]); d_vec_add(in[33], in[33], in[32]);
    d_vec_copy(pre[1], in[33]); d_vec_add(in[35], in[35], in[33]);
    d_vec_add(pre[0], pre[0], in[35]); d_vec_add(in[34], in[34], in[35]);
    d_vec_copy(pre[2], in[34]); d_vec_add(in[38], in[38], in[34]);
    d_vec_add(pre[0], pre[0], in[38]); d_vec_add(in[39], in[39], in[38]);
    d_vec_add(pre[1], pre[1], in[39]); d_vec_add(in[37], in[37], in[39]);
    d_vec_add(pre[0], pre[0], in[37]); d_vec_add(in[36], in[36], in[37]);
    d_vec_copy(pre[3], in[36]); d_vec_add(in[44], in[44], in[36]);
    d_vec_add(pre[0], pre[0], in[44]); d_vec_add(in[45], in[45], in[44]);
    d_vec_add(pre[1], pre[1], in[45]); d_vec_add(in[47], in[47], in[45]);
    d_vec_add(pre[0], pre[0], in[47]); d_vec_add(in[46], in[46], in[47]);
    d_vec_add(pre[2], pre[2], in[46]); d_vec_add(in[42], in[42], in[46]);
    d_vec_add(pre[0], pre[0], in[42]); d_vec_add(in[43], in[43], in[42]);
    d_vec_add(pre[1], pre[1], in[43]); d_vec_add(in[41], in[41], in[43]);
    d_vec_add(pre[0], pre[0], in[41]); d_vec_add(in[40], in[40], in[41]);
    d_vec_copy(pre[4], in[40]); d_vec_add(in[56], in[56], in[40]);
    d_vec_add(pre[0], pre[0], in[56]); d_vec_add(in[57], in[57], in[56]);
    d_vec_add(pre[1], pre[1], in[57]); d_vec_add(in[59], in[59], in[57]);
    d_vec_add(pre[0], pre[0], in[59]); d_vec_add(in[58], in[58], in[59]);
    d_vec_add(pre[2], pre[2], in[58]); d_vec_add(in[62], in[62], in[58]);
    d_vec_add(pre[0], pre[0], in[62]); d_vec_add(in[63], in[63], in[62]);
    d_vec_add(pre[1], pre[1], in[63]); d_vec_add(in[61], in[61], in[63]);
    d_vec_add(pre[0], pre[0], in[61]); d_vec_add(in[60], in[60], in[61]);
    d_vec_add(pre[3], pre[3], in[60]); d_vec_add(in[52], in[52], in[60]);
    d_vec_add(pre[0], pre[0], in[52]); d_vec_add(in[53], in[53], in[52]);
    d_vec_add(pre[1], pre[1], in[53]); d_vec_add(in[55], in[55], in[53]);
    d_vec_add(pre[0], pre[0], in[55]); d_vec_add(in[54], in[54], in[55]);
    d_vec_add(pre[2], pre[2], in[54]); d_vec_add(in[50], in[50], in[54]);
    d_vec_add(pre[0], pre[0], in[50]); d_vec_add(in[51], in[51], in[50]);
    d_vec_add(pre[1], pre[1], in[51]); d_vec_add(in[49], in[49], in[51]);
    d_vec_add(pre[0], pre[0], in[49]); d_vec_add(in[48], in[48], in[49]);
    d_vec_copy(pre[5], in[48]); d_vec_add(in[16], in[16], in[48]);
    d_vec_add(pre[0], pre[0], in[16]); d_vec_add(in[17], in[17], in[16]);
    d_vec_add(pre[1], pre[1], in[17]); d_vec_add(in[19], in[19], in[17]);
    d_vec_add(pre[0], pre[0], in[19]); d_vec_add(in[18], in[18], in[19]);
    d_vec_add(pre[2], pre[2], in[18]); d_vec_add(in[22], in[22], in[18]);
    d_vec_add(pre[0], pre[0], in[22]); d_vec_add(in[23], in[23], in[22]);
    d_vec_add(pre[1], pre[1], in[23]); d_vec_add(in[21], in[21], in[23]);
    d_vec_add(pre[0], pre[0], in[21]); d_vec_add(in[20], in[20], in[21]);
    d_vec_add(pre[3], pre[3], in[20]); d_vec_add(in[28], in[28], in[20]);
    d_vec_add(pre[0], pre[0], in[28]); d_vec_add(in[29], in[29], in[28]);
    d_vec_add(pre[1], pre[1], in[29]); d_vec_add(in[31], in[31], in[29]);
    d_vec_add(pre[0], pre[0], in[31]); d_vec_add(in[30], in[30], in[31]);
    d_vec_add(pre[2], pre[2], in[30]); d_vec_add(in[26], in[26], in[30]);
    d_vec_add(pre[0], pre[0], in[26]); d_vec_add(in[27], in[27], in[26]);
    d_vec_add(pre[1], pre[1], in[27]); d_vec_add(in[25], in[25], in[27]);
    d_vec_add(pre[0], pre[0], in[25]); d_vec_add(in[24], in[24], in[25]);
    d_vec_add(pre[4], pre[4], in[24]); d_vec_add(in[8], in[8], in[24]);
    d_vec_add(pre[0], pre[0], in[8]); d_vec_add(in[9], in[9], in[8]);
    d_vec_add(pre[1], pre[1], in[9]); d_vec_add(in[11], in[11], in[9]);
    d_vec_add(pre[0], pre[0], in[11]); d_vec_add(in[10], in[10], in[11]);
    d_vec_add(pre[2], pre[2], in[10]); d_vec_add(in[14], in[14], in[10]);
    d_vec_add(pre[0], pre[0], in[14]); d_vec_add(in[15], in[15], in[14]);
    d_vec_add(pre[1], pre[1], in[15]); d_vec_add(in[13], in[13], in[15]);
    d_vec_add(pre[0], pre[0], in[13]); d_vec_add(in[12], in[12], in[13]);
    d_vec_add(pre[3], pre[3], in[12]); d_vec_add(in[4], in[4], in[12]);
    d_vec_add(pre[0], pre[0], in[4]); d_vec_add(in[5], in[5], in[4]);
    d_vec_add(pre[1], pre[1], in[5]); d_vec_add(in[7], in[7], in[5]);
    d_vec_add(pre[0], pre[0], in[7]); d_vec_add(in[6], in[6], in[7]);
    d_vec_add(pre[2], pre[2], in[6]); d_vec_add(in[2], in[2], in[6]);
    d_vec_add(pre[0], pre[0], in[2]); d_vec_add(in[3], in[3], in[2]);
    d_vec_add(pre[1], pre[1], in[3]); d_vec_add(in[1], in[1], in[3]);

    // out0 = in[0] + in[1], and pre[0] += in[1]
    d_vec_add(out01[0], in[0], in[1]);
    d_vec_add(pre[0], pre[0], in[1]);

    // out1 = sum_i pre[i] * mask(beta[i])
    // make masks from c_beta
    uint64_t maskv[GFBITS];
    #pragma unroll
    for (int j = 0; j < GFBITS; j++) maskv[j] = ((c_beta[0] >> j) & 1) ? ~0ULL : 0ULL;
    d_vec_mul(out01[1], pre[0], maskv);

    for (int i = 1; i < 6; i++) {
        #pragma unroll
        for (int j = 0; j < GFBITS; j++) maskv[j] = ((c_beta[i] >> j) & 1) ? ~0ULL : 0ULL;
        uint64_t prod[GFBITS];
        d_vec_mul(prod, pre[i], maskv);
        #pragma unroll
        for (int j = 0; j < GFBITS; j++) out01[1][j] ^= prod[j];
    }
}

// ===================================== Kernels functions =========================================

// preprocess: s(bytes) -> recv[64] (uint64 words), little-endian load8 every 8 bytes
// k_preprocess: Pack syndrome bytes 's' into 64 lanes (recv[64]).
// Each thread writes one 64-bit lane (grid-stride) assembling little-endian words.
__global__ void k_preprocess(vec* __restrict__ recv, const unsigned char* __restrict__ s) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < 64; i += stride) {
        uint64_t v = 0;
        #pragma unroll
        for (int b = 7; b >= 0; --b) {
            unsigned char byte = 0;
            size_t off = (size_t)i*8 + b;
            if (off < SYND_BYTES) byte = s[off];  // read s[i*8 + b]
            v <<= 8;
            v |= (uint64_t)byte;
        }
        recv[i] = v;
    }
}

// postprocess: error[64] lanes -> e[SYS_N/8] (pack first SYS_N bits)
// Pack 64 lanes 'error[i]' into byte array 'e'.
// Each lane writes one 64-bit word into e (packed support bits).
// grid-stride loop + a single 64-bit store per lane
__global__ void k_postprocess(unsigned char* __restrict__ e, const vec* __restrict__ error) {
    int stride = blockDim.x * gridDim.x;
    uint64_t* out64 = reinterpret_cast<uint64_t*>(e);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 64; i += stride) {
        out64[i] = error[i];
    }
}

// k_form_error: Build error mask from eval matrix (64 x GFBITS).
// error formation: error[i] = OR_j eval[i][j] ^ allone
// eval64x: 64 rows, each with GFBITS columns (row-major)
// error64: 64 words; bit j set if evaluation at coordinate j is ZERO in GF
__global__ void k_form_error(vec* __restrict__ error, const vec* __restrict__ eval /*[64][GFBITS]*/)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 64; i += stride) {
        const vec* row = eval + i * GFBITS;
        vec r = d_vec_or_reduce(row);      // OR of the GFBITS slices
        r ^= d_vec_setbits(1);             // invert: 1s where value==0
        error[i] = r;
    }
}

// scaling_inv / scaling final step:
// k_and_scale: AND-mask a row-major [64 x GFBITS] matrix with column mask.
// out[i][b] = inv[i][b] & col[i]; used to apply per-coordinate scaling.
// grid-stride over rows; each thread processes a whole row’s 12 bits in a tiny inner loop
__global__ void k_and_scale(vec* __restrict__ out, const vec* __restrict__ inv, const vec* __restrict__ col) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < 64; i += stride) {
        const uint64_t m = col[i];
        const vec* in_row  = inv    + i*GFBITS;
        vec*       out_row = out    + i*GFBITS;
        #pragma unroll
        for (int b = 0; b < GFBITS; ++b) out_row[b] = in_row[b] & m;
    }
}

// k_fft: 64-thread forward FFT over GF in bit-slice form.
// Loads inGFBITS (e.g., irr or locator), does radix conversions + butterflies,
// applies c_scalars/consts/powers, outputs out64x[64*GFBITS].
// One 64xGFBITS FFT; 64 threads cooperate, 1 block
__global__ void k_fft(vec* __restrict__ out64x, vec* __restrict__ inGFBITS)
{
    const int tid = threadIdx.x;
    if (blockIdx.x != 0 || tid >= 64) return;  // single-block transform

    // --- shared working sets ---
    __shared__ uint64_t in_radix[GFBITS];        // size 12
    __shared__ uint64_t cur[64][GFBITS];         // 64*12*8 = 6 KB
    __shared__ uint64_t nxt[64][GFBITS];         // another 6 KB

    // ---- load input, radix_conversions(in) on thread 0 ----
    if (tid < GFBITS) in_radix[tid] = inGFBITS[tid];
    __syncthreads();

    if (tid == 0) {
        // same masks as your current k_fft
        const uint64_t mask[5][2] = {
            {0x8888888888888888ULL, 0x4444444444444444ULL},
            {0xC0C0C0C0C0C0C0C0ULL, 0x3030303030303030ULL},
            {0xF000F000F000F000ULL, 0x0F000F000F000F00ULL},
            {0xFF000000FF000000ULL, 0x00FF000000FF0000ULL},
            {0xFFFF000000000000ULL, 0x0000FFFF00000000ULL}
        };

        // inplace radix conversions + scaling by c_scalars[j]
        for (int j = 0; j <= 4; j++) {
            for (int i = 0; i < GFBITS; i++) {
                for (int k = 4; k >= j; k--) {
                    in_radix[i] ^= (in_radix[i] & mask[k][0]) >> (1 << k);
                    in_radix[i] ^= (in_radix[i] & mask[k][1]) >> (1 << k);
                }
            }
            uint64_t tmp[GFBITS];
            d_vec_mul(tmp, in_radix, c_scalars[j]);
            #pragma unroll
            for (int i=0;i<GFBITS;i++) in_radix[i] = tmp[i];
        }
    }
    __syncthreads();

    // ---- broadcast: each thread builds its row from in_radix ----
    #pragma unroll
    for (int b = 0; b < GFBITS; b++) {
        uint64_t bit = (in_radix[b] >> c_reversal[tid]) & 1ULL;
        cur[tid][b] = d_vec_setbits(bit);
    }
    __syncthreads();

    // ---- butterflies with stage constants (parallel per stage) ----
    int consts_ptr = 0;
    for (int st = 0; st <= 5; st++) {
        int s = 1 << st;

        // compute block start for this thread’s pair
        int blockStart = (tid / (2*s)) * (2*s);
        bool is_lower = (tid - blockStart) < s;

        if (is_lower) {
            int k  = tid;
            int kp = k + s;
            int off = k - blockStart;

            // tmp = cur[k+s] * consts[consts_ptr + off]
            uint64_t tmp[GFBITS];
            d_vec_mul(tmp, cur[kp], c_consts[consts_ptr + off]);

            // out_low  = cur[k]   ^ tmp
            // out_high = cur[k+s] ^ out_low
            #pragma unroll
            for (int b=0;b<GFBITS;b++) {
                uint64_t low  = cur[k][b]  ^ tmp[b];
                uint64_t high = cur[kp][b] ^ low;
                nxt[k][b]  = low;
                nxt[kp][b] = high;
            }
        }
        __syncthreads();

        // swap: copy nxt -> cur in parallel
        #pragma unroll
        for (int b=0;b<GFBITS;b++) cur[tid][b] = nxt[tid][b];
        __syncthreads();

        consts_ptr += s;
    }

    // ---- add powers contribution & write out ----
    #pragma unroll
    for (int b=0;b<GFBITS;b++) {
        uint64_t v = cur[tid][b] ^ c_powers[tid][b];
        out64x[tid*GFBITS + b] = v;
    }
}

// k_fft_tr: 64-thread transpose FFT (inverse pipeline).
// Runs butterflies_tr, 64x64 transpose per slice, broadcast+beta, radix_conv_tr.
// Outputs two GFBITS vectors (out2x) matching the CPU 'fft_tr' layout.
// 64-thread cooperative k_fft_tr (launch as <<<1,64>>>)
__global__ void k_fft_tr(uint64_t* __restrict__ out2x, uint64_t* __restrict__ in64x)
{
    const int tid = threadIdx.x;
    if (blockIdx.x != 0 || tid >= 64) return;

    // ---- shared working sets (two buffers for butterflies) ----
    __shared__ uint64_t cur[64][GFBITS];
    __shared__ uint64_t nxt[64][GFBITS];

    // ---- load input rows in parallel ----
    #pragma unroll
    for (int b = 0; b < GFBITS; b++)
        cur[tid][b] = in64x[tid*GFBITS + b];
    __syncthreads();

    // ================= butterflies_tr (parallel per stage) =================
    // In-place rule (per pair k,k+s):
    //   low  = (a ^ b)
    //   tmp  = low * const
    //   high = b ^ tmp
    //   in[k]   = low
    //   in[k+s] = high
    int consts_ptr = 63;
    for (int st = 5; st >= 0; --st) {
        int s = 1 << st;
        consts_ptr -= s;

        // each lower-half thread in each 2s block computes the pair
        int blockStart = (tid / (2*s)) * (2*s);
        bool is_lower  = (tid - blockStart) < s;

        if (is_lower) {
            int k   = tid;
            int kp  = k + s;
            int off = k - blockStart;

            // low = cur[k] ^ cur[k+s]
            uint64_t low[GFBITS];
            #pragma unroll
            for (int b=0;b<GFBITS;b++) low[b] = cur[k][b] ^ cur[kp][b];

            // tmp = low * consts[consts_ptr + off]
            uint64_t tmp[GFBITS];
            d_vec_mul(tmp, low, c_consts[consts_ptr + off]);

            // high = cur[k+s] ^ tmp
            uint64_t high[GFBITS];
            #pragma unroll
            for (int b=0;b<GFBITS;b++) high[b] = cur[kp][b] ^ tmp[b];

            // write stage output
            #pragma unroll
            for (int b=0;b<GFBITS;b++) { nxt[k][b] = low[b]; nxt[kp][b] = high[b]; }
        }
        __syncthreads();

        // swap: nxt -> cur (parallel copy)
        #pragma unroll
        for (int b=0;b<GFBITS;b++) cur[tid][b] = nxt[tid][b];
        __syncthreads();
    }

    // ================= transpose stage (with bit reversal) =================
    // Parallel across bit-slices (GFBITS threads)
    if (tid < GFBITS) {
        uint64_t buf[64];
        // gather with bit-reversal
        for (int j = 0; j < 64; j++) buf[c_reversal[j]] = cur[j][tid];
        // 64x64 bit-matrix transpose of this bit-slice
        d_transpose_64x64(buf, buf);
        // scatter back
        for (int j = 0; j < 64; j++) cur[j][tid] = buf[j];
    }
    __syncthreads();

    // ================= broadcast + beta accumulation =================
    // Reuse existing device helper that matches CPU reference
    __shared__ uint64_t out01[2][GFBITS];
    if (tid == 0) {
        d_broadcast_and_beta(out01, cur);
    }
    __syncthreads();

    // ================= radix_conversions_tr(out01) =================
    const uint64_t mask[6][2] = {
        {0x2222222222222222ULL, 0x4444444444444444ULL},
        {0x0C0C0C0C0C0C0C0CULL, 0x3030303030303030ULL},
        {0x00F000F000F000F0ULL, 0x0F000F000F000F00ULL},
        {0x0000FF000000FF00ULL, 0x00FF000000FF0000ULL},
        {0x00000000FFFF0000ULL, 0x0000FFFF00000000ULL},
        {0xFFFFFFFF00000000ULL, 0x00000000FFFFFFFFULL}
    };

    for (int j = 5; j >= 0; --j) {
        // scaling needs full-vector multiply: do it once
        if (tid == 0 && j < 5) {
            uint64_t t0[GFBITS], t1[GFBITS];
            d_vec_mul(t0, out01[0], c_scalars2x[j][0]);
            d_vec_mul(t1, out01[1], c_scalars2x[j][1]);
            #pragma unroll
            for (int b = 0; b < GFBITS; b++) { out01[0][b] = t0[b]; out01[1][b] = t1[b]; }
        }
        __syncthreads();

        // per-slice mask/shift scrambles (independent for each b)
        if (tid < GFBITS) {
            uint64_t v0 = out01[0][tid];
            uint64_t v1 = out01[1][tid];

            for (int k = j; k <= 4; k++) {
                v0 ^= (v0 & mask[k][0]) << (1 << k);
                v0 ^= (v0 & mask[k][1]) << (1 << k);
                v1 ^= (v1 & mask[k][0]) << (1 << k);
                v1 ^= (v1 & mask[k][1]) << (1 << k);
            }
            // cross-lane 32-bit mix (still per-slice)
            v1 ^= (v0 & mask[5][0]) >> 32;
            v1 ^= (v1 & mask[5][1]) << 32;

            out01[0][tid] = v0;
            out01[1][tid] = v1;
        }
        __syncthreads();
    }

    // store (any 64 threads can do this, but tid==0 is fine)
    if (tid == 0) {
        #pragma unroll
        for (int b = 0; b < GFBITS; b++) out2x[0*GFBITS + b] = out01[0][b];
        #pragma unroll
        for (int b = 0; b < GFBITS; b++) out2x[1*GFBITS + b] = out01[1][b];
    }
}

// k_sq_rows: Square each of the 64 rows of a [64 x GFBITS] matrix.
__global__ void k_sq_rows(uint64_t* __restrict__ eval64x) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 64; i += stride) {
        uint64_t *row = eval64x + i*GFBITS;
        uint64_t tmp[GFBITS];
        d_vec_sq(tmp, row);
        #pragma unroll
        for (int b=0; b<GFBITS; ++b) row[b] = tmp[b];
    }
}

// k_build_inv: Build multiplicative inverses for all 64 coordinates.
// Prefix products, invert final product, then back-propagate (no extra buffers).
// Build all inverses: given eval[64][GFBITS], produce inv[64][GFBITS]
__global__ void k_build_inv(uint64_t* __restrict__ inv64x, const uint64_t* __restrict__ eval64x) {
    if (blockIdx.x | threadIdx.x) return;

    // prefix: inv[i] = prod_{k=0..i} eval[k]
    // note: write to inv as we go to avoid extra buffers
    // inv[0] = eval[0]
    #pragma unroll
    for (int b=0;b<GFBITS;b++) inv64x[b] = eval64x[b];
    // inv[i] = inv[i-1] * eval[i]
    for (int i=1;i<64;i++) {
        d_vec_mul(inv64x + i*GFBITS, inv64x + (i-1)*GFBITS, eval64x + i*GFBITS);
    }

    // tmp = (inv[63])^{-1}
    uint64_t tmp[GFBITS];
    d_vec_inv(tmp, inv64x + 63*GFBITS);

    // back pass:
    // for i=62..0:
    //   inv[i+1] = tmp * inv[i]
    //   tmp      = tmp * eval[i+1]
    for (int i=62;i>=0;i--) {
        uint64_t t[GFBITS];
        d_vec_mul(t, tmp, inv64x + i*GFBITS);
        #pragma unroll
        for (int b=0;b<GFBITS;b++) inv64x[(i+1)*GFBITS + b] = t[b];
        d_vec_mul(tmp, tmp, eval64x + (i+1)*GFBITS);
    }
    // inv[0] = tmp
    #pragma unroll
    for (int b=0;b<GFBITS;b++) inv64x[b] = tmp[b];
}

// k_benes: Apply Benes network permutation (rev=1 forward, rev=0 inverse).
// Loads condition bits from 'bits', transposes them, and applies layered swaps.
// A fixed sequence of conditional swaps (butterflies) that permutes data (syndrome) according to a secret permutation.
// Shuffling and unshuffling bits in a structured way
__global__ void k_benes(uint64_t* __restrict__ r, const unsigned char* __restrict__ bits, int rev)
{
    // One cooperative block (threads within the block scale the work)
    if (blockIdx.x != 0) return;

    const unsigned char* cond_ptr;
    int inc;
    if (rev == 0) { inc = 256;  cond_ptr = bits; }
    else          { inc = -256; cond_ptr = bits + (2*GFBITS - 2) * 256; }

    __shared__ uint64_t cond64[64];  // reuse for 32/64-entry stages

    // ===== stage 1 =====
    if (threadIdx.x == 0) d_transpose_64x64(r, r);
    __syncthreads();

    for (int low = 0; low <= 5; ++low) {
        // load 64 × 32-bit conds in parallel
        for (int i = threadIdx.x; i < 64; i += blockDim.x)
            cond64[i] = (uint64_t)d_load4(cond_ptr + i*4);
        __syncthreads();

        // transpose conds (single thread) and layer in parallel
        if (threadIdx.x == 0) d_transpose_64x64(cond64, cond64);
        __syncthreads();

        d_layer_parallel(r, cond64, low, threadIdx.x, blockDim.x);
        __syncthreads();

        cond_ptr += inc;
    }

    // ===== stage 2 =====
    if (threadIdx.x == 0) d_transpose_64x64(r, r);
    __syncthreads();

    for (int low = 0; low <= 5; ++low) {
        // load 32 × 64-bit conds in parallel
        for (int i = threadIdx.x; i < 32; i += blockDim.x)
            cond64[i] = d_load8(cond_ptr + i*8);
        __syncthreads();

        d_layer_parallel(r, cond64, low, threadIdx.x, blockDim.x);
        __syncthreads();

        cond_ptr += inc;
    }
    for (int low = 4; low >= 0; --low) {
        for (int i = threadIdx.x; i < 32; i += blockDim.x)
            cond64[i] = d_load8(cond_ptr + i*8);
        __syncthreads();

        d_layer_parallel(r, cond64, low, threadIdx.x, blockDim.x);
        __syncthreads();

        cond_ptr += inc;
    }

    // ===== stage 3 =====
    if (threadIdx.x == 0) d_transpose_64x64(r, r);
    __syncthreads();

    for (int low = 5; low >= 0; --low) {
        for (int i = threadIdx.x; i < 64; i += blockDim.x)
            cond64[i] = (uint64_t)d_load4(cond_ptr + i*4);
        __syncthreads();

        if (threadIdx.x == 0) d_transpose_64x64(cond64, cond64);
        __syncthreads();

        d_layer_parallel(r, cond64, low, threadIdx.x, blockDim.x);
        __syncthreads();

        cond_ptr += inc;
    }

    if (threadIdx.x == 0) d_transpose_64x64(r, r);
}

// k_synd_cmp: Compare two (2 x GFBITS) vectors (s0 vs s1) on device.
// Grid-stride OR of differences -> sets out_ok=1 if equal (uses atomic flags).
__global__ void k_synd_cmp(const uint64_t* __restrict__ s0_2x,
                           const uint64_t* __restrict__ s1_2x,
                           unsigned short* __restrict__ out_ok,
                           unsigned int* __restrict__ d_flag,
                           unsigned int* __restrict__ d_done)
{
    // grid-stride OR over the 2*GFBITS words (e.g., 24)
    uint64_t local = 0;
    int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < 2*GFBITS; idx += stride) {
        local |= (s0_2x[idx] ^ s1_2x[idx]);
    }

    // If any bit differs in this thread's chunk, set the global flag to 1
    if (local) atomicOr(d_flag, 1u);

    __syncthreads(); // intra-block

    // Last block writes out_ok (no host round-trip needed)
    if (threadIdx.x == 0) {
        unsigned int ticket = atomicAdd(d_done, 1u);  // returns old value
        if (ticket == gridDim.x - 1) {
            *out_ok = (*d_flag == 0u) ? 1 : 0;
        }
    }
}

// Compute w0 (popcount over all 4096 support bits) and w1 (popcount over first SYS_N bits)
__global__ void k_weight_check(const uint64_t* __restrict__ Err64,   // 64 lanes
                               const unsigned char* __restrict__ E,  // packed bits (4096/8 bytes)
                               unsigned int* __restrict__ out_w0,
                               unsigned int* __restrict__ out_w1)
{
    unsigned int local_w0 = 0;

    // Sum popcounts over 64 x 64-bit words (Err64[0..63])
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < 64; i += stride) {
        local_w0 += __popcll(Err64[i]);
    }

    // Sum popcounts over first SYS_N bits of E (mask final byte)
    int total_bytes = (SYS_N + 7) >> 3;           // ceil(SYS_N / 8)
    int last_bits   =  SYS_N        & 7;          // remaining bits in last byte (0..7)
    unsigned int local_w1 = 0;

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < total_bytes; i += stride) {
        unsigned char b = E[i];
        if (last_bits && (i == total_bytes - 1)) {
            b &= (unsigned char)((1u << last_bits) - 1u);
        }
        local_w1 += __popc((unsigned int)b);
    }

    // Atomically accumulate to outputs
    if (local_w0) atomicAdd(out_w0, local_w0);
    if (local_w1) atomicAdd(out_w1, local_w1);
}

// ============================ Host decrypt ====================================
extern "C"
// decrypt: Orchestrates Niederreiter decryption on GPU.
// H2D: copy s, Benes bits, irr. Pipelines:
//   preprocess → benes(fwd) → FFT(irr) → sq_rows → build_inv
//   → and_scale(recv) → fft_tr → BM (CPU) → FFT(locator)
//   → form_error → and_scale(error) → fft_tr + compare
//   → benes(inv) → postprocess (pack e) → weight check.
// Returns 0 on success (syndrome+weight OK), 1 on failure.

int decrypt(unsigned char *e, const unsigned char *sk, const unsigned char *s)
{
    // Load __constant__ FFT tables (once per process)
    fft_tables_cuda_init();

    using vec = uint64_t;

    // ============= HOST WORK BUFFERS ============
    vec        h_irr_int[GFBITS];                // filled by irr_load(...)
    uint64_t   h_s_priv[2][GFBITS];              // filled from d_s_priv
    uint64_t   locator[GFBITS];                  // filled by bm(...)
    unsigned short h_ok = 0;                     // overwritten by cudaMemcpy from d_ok
    unsigned char  e_full[(1<<GFBITS)/8];        // filled by cudaMemcpy from d_e
    unsigned int   h_w0 = 0, h_w1 = 0;           // overwritten by cudaMemcpy from d_w0/d_w1


    // ===================== ALL DEVICE ALLOS UP FRONT =====================
    unsigned char *d_s=nullptr, *d_bits=nullptr, *d_e=nullptr;
    vec *d_recv=nullptr, *d_inv=nullptr, *d_scaled=nullptr, *d_eval=nullptr, *d_error=nullptr,
        *d_s_priv=nullptr, *d_s_priv_cmp=nullptr, *d_irr=nullptr, *d_locator=nullptr;
    unsigned short *d_ok=nullptr;
    unsigned int *d_flag = nullptr, *d_done = nullptr;
    unsigned int *d_w0=nullptr, *d_w1=nullptr;

    // ===================== DEVICE ALLOCATIONS =====================
    cudaMalloc(&d_s,       SYND_BYTES);
    cudaMalloc(&d_bits,    COND_BYTES);
    cudaMalloc(&d_e,       ((1<<GFBITS)/8));

    cudaMalloc(&d_recv,        64*sizeof(vec));
    cudaMalloc(&d_inv,   64*GFBITS*sizeof(vec));
    cudaMalloc(&d_scaled,64*GFBITS*sizeof(vec));
    cudaMalloc(&d_eval,  64*GFBITS*sizeof(vec));
    cudaMalloc(&d_error,       64*sizeof(vec));
    cudaMalloc(&d_s_priv,   2*GFBITS*sizeof(vec));
    cudaMalloc(&d_s_priv_cmp, 2*GFBITS*sizeof(vec));
    cudaMalloc(&d_irr,          GFBITS*sizeof(vec));
    cudaMalloc(&d_locator,      GFBITS*sizeof(vec));

    cudaMalloc(&d_ok,   sizeof(unsigned short));
    cudaMalloc(&d_flag, sizeof(unsigned int));
    cudaMalloc(&d_done, sizeof(unsigned int));
    cudaMalloc(&d_w0,   sizeof(unsigned int));
    cudaMalloc(&d_w1,   sizeof(unsigned int));

    // ===================== ALL INITIAL H2D COPIES =====================
    // s -> device
    cudaMemcpy(d_s, s, SYND_BYTES, cudaMemcpyHostToDevice);

    // Benes bits from secret key
    cudaMemcpy(d_bits, sk + IRR_BYTES, COND_BYTES, cudaMemcpyHostToDevice);

    // irr (bitsliced) -> device
    irr_load(reinterpret_cast<uint64_t*>(h_irr_int), sk);
    cudaMemcpy(d_irr, h_irr_int, GFBITS*sizeof(vec), cudaMemcpyHostToDevice);

    // =================== Timing Setup ===============================
    time_t now = time(0);
    tm *ltm = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H:%M:%S", ltm);

    std::ostringstream oss;
    oss << "results_demo/decrypt_" << timestamp << ".csv";
    std::string filename = oss.str();

    FILE *log_file = fopen(filename.c_str(), "w");
    if (!log_file) {
        fprintf(stderr, "Failed to open result file: %s\n", filename.c_str());
        // still continue the actual decrypt once with num_blocks=1
    } else {
        fprintf(log_file, "num_blocks,trial,time_ms,throughput\n");
    }

    const float total_items = float(SYS_N) / 8.0f; // bytes processed

    // ==================== Num_Blocks Loop (Batch)==============================
    for (int num_blocks = 1; num_blocks <= 32; num_blocks *= 2){
        printf("\n===== DECRYPT: testing with %d blocks =====\n", num_blocks);

        float total_ms = 0.0f, total_throughput = 0.0f;

        int trial;
        // ============= Trials Loop =============
        for (trial = 1; trial <= 10; ++trial){
            //Reset every trials
            cudaMemset(d_ok,   0, sizeof(unsigned short));
            cudaMemset(d_flag, 0, sizeof(unsigned int));
            cudaMemset(d_done, 0, sizeof(unsigned int));
            cudaMemset(d_w0, 0, sizeof(unsigned int));
            cudaMemset(d_w1, 0, sizeof(unsigned int));

            // timer
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            // ===================== PIPELINE =====================
            // Preprocess: syndrome -> recv
            // 1) take the ciphertext syndrome and load it into 64 lanes on the GPU
            k_preprocess<<<num_blocks,64>>>(d_recv, d_s);

            //2) Benes Network: like a secret shuffle that rearranges the data according to the private key
            // Benes forward (Only 1 block)
            k_benes<<<1,64>>>((uint64_t*)d_recv, d_bits, /*rev=*/1);

            //3) Evaluate the secret polynomial and build inverses, which prepare the data for solving the key equation
            // FFT(irr), square rows, build inv (1 block only)
            k_fft<<<1,64>>>(d_eval, d_irr);
            k_sq_rows<<<num_blocks,64>>>(d_eval);
            k_build_inv<<<1,1>>>(d_inv, d_eval);

            //4) Scale the syndrome and run an inverse FFT (fft_tr) to get the private-domain syndrome
            // scaled = inv & recv
            k_and_scale<<<num_blocks, 64>>>(d_scaled, d_inv, d_recv);

            // fft_tr(s_priv, scaled) on GPU (1 block)
            k_fft_tr<<<1,64>>>(d_s_priv, d_scaled);

            // ---------- BM on CPU -> to Device ----------
            // 5) An algorithm that computes the error locator polynomial from the syndrome
            // Tells which positions are flipped during encryption
            cudaMemcpy(h_s_priv, d_s_priv, 2*GFBITS*sizeof(uint64_t), cudaMemcpyDeviceToHost);
            bm(locator, h_s_priv);
            cudaMemcpy(d_locator, locator, GFBITS*sizeof(uint64_t), cudaMemcpyHostToDevice);

            //6) Do another FFT to mark error positions, and form the error vector
            // FFT(eval, locator) 1 block only
            k_fft<<<1,64>>>(d_eval, d_locator);

            // form error
            k_form_error<<<num_blocks,64>>>(d_error, d_eval);
            cudaDeviceSynchronize();

            // 7) Scale it with the inv we built, makes sure error vector is in right domain b4 transform
            // scaled = inv & error
            k_and_scale<<<num_blocks, 64>>>(d_scaled, d_inv, d_error);

            // 8) Turn scaled error back to syndrome form, then compare it with original syndrome
            k_fft_tr<<<1,64>>>(d_s_priv_cmp, d_scaled);
            k_synd_cmp<<<num_blocks,64>>>((const uint64_t*)d_s_priv, (const uint64_t*)d_s_priv_cmp, d_ok, d_flag, d_done);

            // 9) undo the Benes permutation, pack the error bits, and check if the weight matches
            // inverse Benes on error (Only 1 block)
            k_benes<<<1,64>>>((uint64_t*)d_error, d_bits, /*rev=*/0);

            // postprocess: pack error to bytes
            k_postprocess<<<num_blocks,64>>>(d_e, d_error);

            // weight check
            k_weight_check<<<num_blocks,64>>>((uint64_t*)d_error, d_e, d_w0, d_w1);

            // stop timer
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            float seconds = ms / 1000.0f;
            float throughput = (total_items * num_blocks) / (seconds > 0 ? seconds : 1e-9f); //Bytes processed / sec

            printf("Trial %d: Time = %.6f ms | Throughput = %.2f items/s\n", trial, ms, throughput);
            if (log_file) fprintf(log_file, "%d,%d,%.6f,%.2f\n", num_blocks, trial, ms, throughput);

            total_ms += ms;
            total_throughput += throughput;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            // bring back flags and e for invariance check
            cudaMemcpy(&h_ok, d_ok, sizeof(unsigned short), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_w0, d_w0, sizeof(unsigned int),  cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_w1, d_w1, sizeof(unsigned int),  cudaMemcpyDeviceToHost);

            cudaMemcpy(e_full, d_e, (1<<GFBITS)/8, cudaMemcpyDeviceToHost);
        }//trials loop

        float avg_ms = total_ms / float(trial-1);
        float avg_throughput = total_throughput / float(trial-1);
        printf("Average for %d blocks: Time = %.6f ms | Throughput = %.2f items/s\n", num_blocks, avg_ms, avg_throughput);
        if (log_file) fprintf(log_file, "%d,avg,%.6f,%.2f\n", num_blocks, avg_ms, avg_throughput);
        
    }//NumBlocks loop

    if (log_file) fclose(log_file);

    // set caller's e[] from the first (reference) run
    memcpy(e, e_full, SYS_N/8);

    // final correctness flags from the last run (should be stable)
    uint16_t check = ((h_w0 ^ SYS_T) | (h_w1 ^ SYS_T));
    check -= 1; check >>= 15;
    uint16_t check_weight = check;

    // Bring back the 1-bit result
    cudaMemcpy(&h_ok, d_ok, sizeof(unsigned short), cudaMemcpyDeviceToHost);
    uint16_t check_synd = h_ok; // 1==equal

    // ---------- Results (compare with encrypt) ----------
    {
        int k;
        printf("\nDecrypt e: positions");
        for (k = 0; k < SYS_N; ++k)
            if (e[k/8] & (1u << (k & 7)))
                printf(" %d", k);
        printf("\n");
    }
    {
        printf("\nSyndrome s (Decrypt): ");
        for (int i = 0; i < SYND_BYTES; i++) {
            printf("%02X", s[i]);
            if ((i + 1) % 16 == 0) printf("\n");
            else printf(" ");
        }
        printf("\n");
    }

    unsigned char ret = 1 - (check_synd & check_weight); // 0=success, 1=failure

    // ===================== cudaFree =====================
    cudaFree(d_s); cudaFree(d_bits); cudaFree(d_e);
    cudaFree(d_recv); cudaFree(d_inv); cudaFree(d_scaled); cudaFree(d_eval); cudaFree(d_error);
    cudaFree(d_s_priv); cudaFree(d_s_priv_cmp); cudaFree(d_irr); cudaFree(d_locator);
    cudaFree(d_ok); cudaFree(d_flag); cudaFree(d_done); cudaFree(d_w0); cudaFree(d_w1);

    return ret;
}
// WHY some kernels like k_benes, k_fft and k_fft_tr only runs 1 block?
// k_benes: Needs all 64 lanes in shared memory at once to perform layered swaps. Running more than 64 threads per stage is redundant.
// k_fft: Each thread corresponds to one lane (or one element of the support). The transform has fixed size 64, so launching multiple blocks just duplicates work

// crypto_kem_dec: NIST KEM decapsulation wrapper.
// Calls decrypt(e, sk, c). Builds preimage with success mask (m), chooses s or e
// per NIST spec, appends c (syndrome), then SHAKE256 → shared secret 'key'.
int crypto_kem_dec(unsigned char *key, const unsigned char *c,const unsigned char *sk)
{
    int i;
    unsigned char ret_decrypt = 0;
    uint16_t m;

    unsigned char e[SYS_N/8];
    unsigned char preimage[1 + SYS_N/8 + SYND_BYTES];
    unsigned char *x = preimage;
    const unsigned char *s = sk + 40 + IRR_BYTES + COND_BYTES;

    /* Niederreiter decrypt: returns 0 on success, 1 on failure */
    ret_decrypt = decrypt(e, sk + 40, c);

    /* m = 0xFF on success, 0x00 on failure */
    m  = ret_decrypt;
    m -= 1;
    m >>= 8;

    *x++ = (unsigned char)(m & 1);

    for (i = 0; i < SYS_N/8; i++)
        *x++ = (unsigned char)(((~m) & s[i]) | (m & e[i]));

    for (i = 0; i < SYND_BYTES; i++)
        *x++ = c[i];

    crypto_hash_32b(key, preimage, sizeof(preimage));

    return 0;
}
