// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <curand_kernel.h> 
#include <ctime> 
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <set>

// Include associated header file.
#include "../include/cuda_kernel.cuh"
#include "../include/params.h"
#include "../include/crypto_kem.h"
#include "../include/rng.h"
#include "../include/uint16_sort.h"
#include "../include/crypto_hash.h"

//Encrypt function
void encrypt(unsigned char *s, const unsigned char *pk, unsigned char *e){
    // Create timestamp string
    time_t now = time(0);
    tm *ltm = localtime(&now);

    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H:%M:%S", ltm);

    // Construct file path
    std::ostringstream oss;
    oss << "results_demo/encrypt_" << timestamp << ".csv";
    std::string filename = oss.str();

    FILE *log_file = fopen(filename.c_str(), "w");
    if (!log_file) {
        fprintf(stderr, "Failed to open result file: %s\n", filename.c_str());
        return;
    }
    fprintf(log_file, "num_blocks,trial,time_ms,throughput\n");

    float total_items = SYS_N / 8;

    //Generate buf_bytes once only
    unsigned char buf_bytes[SYS_T * 2 * sizeof(uint16_t)];
    randombytes(buf_bytes, sizeof(buf_bytes));

    // Warmup
    warmup_kernel<<<1, SYS_T * 2>>>();

    // Flag to track if we've already captured the actual results
    bool results_captured = false;

    //Benchmarking
    for (int num_blocks = 1; num_blocks <= 32; num_blocks *= 2) {
        printf("\n===== ENCRYPT: Testing with %d blocks =====\n", num_blocks);

        float total_ms = 0.0f;
        float total_throughput = 0.0f;

        int trial;
        for (trial = 1; trial <= 10; ++trial) {
            //Allocations
            uint16_t *d_nums, *d_ind;
            uint64_t *d_val, *d_e_int;
            uint64_t e_int_host[(SYS_N + 63) / 64];
            int *d_count, *d_has_duplicates;
            unsigned char *d_e, *d_bytes, *d_s, *d_pk;
            
            cudaMalloc(&d_nums, sizeof(uint16_t) * SYS_T * 2);
            cudaMalloc(&d_bytes, sizeof(unsigned char) * SYS_T * 2 * sizeof(uint16_t));
            cudaMalloc(&d_ind, sizeof(uint16_t) * SYS_T);
            //int total_elements = SYS_T * num_blocks;
            cudaMalloc(&d_val, sizeof(uint64_t) * SYS_T);
            cudaMalloc(&d_e_int, sizeof(uint64_t) * ((SYS_N + 63) / 64));
            cudaMalloc(&d_count, sizeof(int));
            cudaMalloc(&d_has_duplicates, sizeof(int));
            cudaMalloc(&d_e, sizeof(unsigned char) * (SYS_N / 8));
            cudaMalloc(&d_s, SYND_BYTES);
            cudaMalloc(&d_pk, PK_NROWS * PK_ROW_BYTES);

            cudaMemset(d_count, 0, sizeof(int));
            cudaMemset(d_has_duplicates, 0, sizeof(int));
            cudaMemset(d_e, 0, sizeof(unsigned char) * (SYS_N / 8));
            //cudaMemset(d_s, 0, SYND_BYTES);
            cudaMemset(d_val, 0, sizeof(uint64_t) * SYS_T);

            // Timer
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            // Copy input data
            cudaMemcpy(d_bytes, buf_bytes, sizeof(unsigned char) * SYS_T * 2 * sizeof(uint16_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_pk, pk, PK_NROWS * PK_ROW_BYTES, cudaMemcpyHostToDevice);
            
            // gen_e
            load_gf_kernel<<<num_blocks, SYS_T * 2>>>(d_bytes, d_nums);

            filter_valid_indices<<<num_blocks, SYS_T * 2>>>(d_nums, d_ind, d_count);

            int threads_per_block = SYS_T; 
            int shared_memory_bytes = threads_per_block * sizeof(uint16_t);

            bitonic_sort_kernel<<<1, threads_per_block, shared_memory_bytes>>>(d_ind, SYS_T);

            check_duplicates<<<num_blocks, dim3(SYS_T, SYS_T)>>>(d_ind, d_has_duplicates);

            int has_duplicates;
            cudaMemcpy(&has_duplicates, d_has_duplicates, sizeof(int), cudaMemcpyDeviceToHost);

            if (!has_duplicates) {
                compute_val<<<num_blocks, SYS_T>>>(d_ind, d_val);

                int err_blocks = ((SYS_N + 63)/64 + SYS_T - 1) / SYS_T; 
                set_bits<<<err_blocks, SYS_T>>>(d_ind, d_val, d_e_int);
                cudaDeviceSynchronize();
            }

            if(!results_captured){
                cudaMemcpy(e_int_host, d_e_int, sizeof(e_int_host), cudaMemcpyDeviceToHost);

                //Store back to e
                for (int i = 0; i < (SYS_N + 63) / 64 - 1; i++) {
                    store8(e + i * 8, e_int_host[i]);
                }
                for (int j = 0; j < (SYS_N % 64); j += 8) {
                    e[((SYS_N + 63) / 64 - 1) * 8 + j / 8] = (e_int_host[(SYS_N + 63) / 64 - 1] >> j) & 0xFF;
                }
            }

            // syndrome
            cudaMemcpy(d_s, e, SYND_BYTES, cudaMemcpyHostToDevice);
            cudaMemcpy(d_e, e, SYS_N / 8, cudaMemcpyHostToDevice);
            int syndrome_threads_per_block = 256;
            int blocks_needed = (PK_NROWS + syndrome_threads_per_block - 1) / syndrome_threads_per_block;
            syndrome_kernel_vecstyle<<<blocks_needed, syndrome_threads_per_block>>>(d_s, d_pk, d_e);

            // IMPORTANT: Only capture results the first time through the loop if not the encrypt e positions will be wrong
            // This ensures we always use the same configuration for the actual output
            if (!results_captured) {              
                cudaMemcpy(s, d_s, SYND_BYTES, cudaMemcpyDeviceToHost);  
                results_captured = true;
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            float seconds = ms / 1000.0f;
            float throughput = (total_items * num_blocks) / (seconds > 0 ? seconds : 1e-9f);;

            printf("Trial %d: Time = %.6f ms | Throughput = %.2f items/sec\n", trial, ms, throughput);
            fprintf(log_file, "%d,%d,%.6f,%.2f\n", num_blocks, trial, ms, throughput);

            total_ms += ms;
            total_throughput += throughput;

            //Cleanup
            cudaFree(d_nums); cudaFree(d_ind); cudaFree(d_count); cudaFree(d_has_duplicates);
            cudaFree(d_e); cudaFree(d_val); cudaFree(d_bytes); cudaFree(d_s); cudaFree(d_pk);
            cudaFree(d_e_int); //cudaFree(d_e_extra);
            cudaEventDestroy(start); cudaEventDestroy(stop);
        }

        float avg_ms = total_ms / (float) (trial-1);
        float avg_throughput = total_throughput / (float) (trial-1);
        printf("Average for %d blocks: Time = %.6f ms | Throughput = %.2f items/sec\n", num_blocks, avg_ms, avg_throughput);
        fprintf(log_file, "%d,avg,%.6f,%.2f\n", num_blocks, avg_ms, avg_throughput);
    }

    fclose(log_file);

    //Print encrypt e positions
    int k;
    printf("\nEncrypt e: positions");
    for (k=0; k < SYS_N; ++k)
        if (e[k/8] & (1 << (k&7)))
            printf(" %d", k);
    printf("\n");

    //Print syndrome
    printf("\nSyndrome s (Encrypt): ");
    for (int i = 0; i < SYND_BYTES; i++) {
        printf("%02X", s[i]);
        if ((i + 1) % 16 == 0) printf("\n");
        else printf(" ");
    }
    printf("\n");
}

__device__ uint16_t load_gf(const unsigned char *src) {
    uint16_t a = src[1];
    a <<= 8;
    a |= src[0];
    return a & GFMASK;
}

__device__ uint16_t crypto_uint16_smaller_mask(uint16_t x, uint16_t y) {
    uint16_t z = x - y;
    z ^= (x ^ y) & (z ^ x ^ (1 << 15));
    return z >> 15;
}

__device__ uint16_t uint16_is_smaller_declassify(uint16_t t, uint16_t u) {
    return crypto_uint16_smaller_mask(t, u);
}

__device__ uint32_t crypto_uint32_equal_mask(uint32_t x, uint32_t y) {
    return ~(x ^ y) ? 0xFFFFFFFF : 0;
}

__device__ uint32_t uint32_is_equal_declassify(uint32_t t, uint32_t u) {
    return crypto_uint32_equal_mask(t, u);
}

// __device__ unsigned char same_mask(uint16_t x, uint16_t y) {
//     uint32_t mask = x ^ y;
//     mask -= 1;
//     mask >>= 31;
//     return -mask & 0xFF;
// }


__global__ void load_gf_kernel(unsigned char *src, uint16_t *nums) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < SYS_T * 2) {
        nums[i] = load_gf(src + i * 2);
    }
}


__global__ void filter_valid_indices(uint16_t *nums, uint16_t *ind, int *count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < SYS_T * 2) {
        if (uint16_is_smaller_declassify(nums[i], SYS_N)) {
            int pos = atomicAdd(count, 1);
            if (pos < SYS_T) {
                ind[pos] = nums[i];
            }
        }
    }
}


 __global__ void check_duplicates(uint16_t *ind, int *has_duplicates) {
     int i = threadIdx.x;
     //int j = threadIdx.y;

     if (i > 0 && i < SYS_T) {
         if (uint32_is_equal_declassify(ind[i-1], ind[i])) {
             atomicExch(has_duplicates, 1);
         }
     }
 }


__global__ void compute_val(uint16_t *d_ind, uint64_t *d_val){
    int j = threadIdx.x; // Each thread corresponds to a `val[j]`

    if (j < SYS_T) {
        d_val[j] = 1ULL << (d_ind[j] & 63); // Compute bit shift for each index
        //printf("d_ind[%d] = %u, d_val[%d] = 0x%016llx\n", j, d_ind[j], j, d_val[j]);
    }
}


__global__ void set_bits(uint16_t *d_ind, uint64_t *d_val, uint64_t *d_e_int) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Each thread works on one `e[i]`
    if (i >= (SYS_N + 63) / 64){
        //printf("Failed to run set_bits");
        return;
    }

    d_e_int[i] = 0; //Initialize 'e[i]'

    //uint64_t result = 0;
    for (int j = 0; j < SYS_T; j++) {
        uint64_t mask = i ^ (d_ind[j] >> 6);
        mask -= 1;
        mask >>= 63;
        mask = -mask;

        d_e_int[i] |= d_val[j] & mask;
    }
}

__global__ void bitonic_sort_kernel(uint16_t* data, int num_elements) {
    extern __shared__ uint16_t shared[];

    int tid = threadIdx.x;
    if (tid >= num_elements) return;  // Only use threads for actual data

    // Load from global to shared memory
    shared[tid] = data[tid];
    __syncthreads();

    // Bitonic sort on shared memory
    for (int k = 2; k <= num_elements; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            if (ixj > tid && ixj < num_elements) {
                bool ascending = ((tid & k) == 0);
                uint16_t a = shared[tid];
                uint16_t b = shared[ixj];
                if ((ascending && a > b) || (!ascending && a < b)) {
                    shared[tid] = b;
                    shared[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    // Write back to global memory
    data[tid] = shared[tid];
}


__global__ void warmup_kernel() {
    // Empty kernel, just uses threads for GPU initialization
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // No computation, just thread utilization
}

static inline void store8(unsigned char *out, uint64_t in)
{
    out[0] = (in >> 0x00) & 0xFF;
    out[1] = (in >> 0x08) & 0xFF;
    out[2] = (in >> 0x10) & 0xFF;
    out[3] = (in >> 0x18) & 0xFF;
    out[4] = (in >> 0x20) & 0xFF;
    out[5] = (in >> 0x28) & 0xFF;
    out[6] = (in >> 0x30) & 0xFF;
    out[7] = (in >> 0x38) & 0xFF;
}

// read 64 bits from possibly unaligned address
__device__ inline uint64_t ld64_unaligned(const uint8_t* p) {
    uint64_t v = 0;
    // little-endian assemble
    #pragma unroll
    for (int k = 0; k < 8; ++k) v |= (uint64_t)p[k] << (8*k);
    return v;
}

// read 32 bits from possibly unaligned address
__device__ inline uint32_t ld32_unaligned(const uint8_t* p) {
    uint32_t v = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) v |= (uint32_t)p[k] << (8*k);
    return v;
}

// atomic XOR a single bit in s, but operate on a 32-bit word (CUDA requirement)
__device__ inline void atomic_xor_bit(uint8_t* s, int bit_index) {
    const int byte_index = bit_index >> 3;      // /8
    const int bit_in_byte = bit_index & 7;      // %8
    uint32_t* s32 = reinterpret_cast<uint32_t*>(s);

    const int word_index  = byte_index >> 2;    // /4
    const int byte_in_word = byte_index & 3;    // %4
    const uint32_t mask = ((uint32_t)(1u << bit_in_byte)) << (8 * byte_in_word);

    atomicXor(&s32[word_index], mask);
}

__global__ void syndrome_kernel_vecstyle(
    uint8_t* __restrict__ s,          // length SYND_BYTES, already initialized with e[0:SYND_BYTES]
    const uint8_t* __restrict__ pk,   // PK_NROWS * PK_ROW_BYTES
    const uint8_t* __restrict__ e    // full e (SYS_N/8 bytes); we’ll use e + SYND_BYTES
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= PK_NROWS) return;

    // vec: e_ptr = (uint64_t*)(e + SYND_BYTES)
    const uint8_t* e_tail = e + ((PK_NROWS + 7) / 8);  // == SYND_BYTES
    const uint8_t* row    = pk + (size_t)i * PK_ROW_BYTES;

    const int full64 = PK_NCOLS / 64;      // for your params this is 42
    const int rem    = PK_NCOLS & 63;      // for your params this is 32

    uint64_t b = 0;

    // XOR-accumulate 64-bit chunks (mirrors vec: b ^= pk64[j] & e64[j])
    for (int j = 0; j < full64; ++j) {
        const uint8_t* row_j = row     + (size_t)j * 8;
        const uint8_t* e_j   = e_tail  + (size_t)j * 8;
        b ^= (ld64_unaligned(row_j) & ld64_unaligned(e_j));
    }

    // 32-bit tail (vec does a final 32-bit AND/XOR via casts)
    if (rem >= 32) {
        const uint8_t* row_tail = row    + (size_t)full64 * 8;
        const uint8_t* e_tail32 = e_tail + (size_t)full64 * 8;
        b ^= (uint64_t)(ld32_unaligned(row_tail) & ld32_unaligned(e_tail32));
        // (rem is 32 for your params; if you ever had >32, add a masked second 32-bit read)
    }

    // vec’s parity fold:
    b ^= b >> 32;
    b ^= b >> 16;
    b ^= b >> 8;
    b ^= b >> 4;
    b ^= b >> 2;
    b ^= b >> 1;
    b &= 1;

    if (b) {
        atomic_xor_bit(s, i);  // s[i/8] ^= (1<<(i%8)) but done atomically as a 32-bit op
    }
}

__global__ void syndrome_kernel(unsigned char *s, const unsigned char *pk, const unsigned char *e){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= PK_NROWS) return;  // Ensure we don't exceed bounds

    const uint64_t *pk_ptr = (const uint64_t *)(pk + i * PK_ROW_BYTES);
    const uint64_t *e_ptr  = (const uint64_t *)e;

    uint64_t b = 0;
    for (int j = 0; j < PK_NCOLS / 64; j++)
        b ^= pk_ptr[j] & e_ptr[j];

    b ^= ((uint32_t *)&pk_ptr[PK_NCOLS / 64])[0] & ((uint32_t *)&e_ptr[PK_NCOLS / 64])[0];
    b ^= b >> 32;
    b ^= b >> 16;
    b ^= b >> 8;
    b ^= b >> 4;
    b ^= b >> 2;
    b ^= b >> 1;
    b &= 1;

    if (b){
        // Safely do atomic XOR on s[i / 8]
        int byte_index = i / 8;
        int bit_offset = i % 8;

        int word_index = byte_index / 4;       // each uint32_t has 4 bytes
        int byte_in_word = byte_index % 4;

        uint32_t mask = ((uint32_t)(1 << bit_offset)) << (8 * byte_in_word);

        atomicXor(((uint32_t*)s) + word_index, mask);
    }
}

int crypto_kem_enc_new(unsigned char *c, unsigned char *key, const unsigned char *pk)
{
    // e: error vector (SYS_N bits => SYS_N/8 bytes)
    unsigned char e[SYS_N/8];
    // one_ec: 1 || e || c  (first byte set to 1, rest zeroed)
    unsigned char one_ec[1 + SYS_N/8 + SYND_BYTES] = {1};

    // Run your CUDA encrypt to fill c (syndrome) and e (error vector)
    // Note: your encrypt writes: s -> c (SYND_BYTES), e -> error vector
    memset(e, 0, sizeof(e));     // good hygiene; encrypt should set bits anyway
    encrypt(c, pk, e);

    // Concatenate: 1 || e || c
    memcpy(one_ec + 1, e, SYS_N/8);
    memcpy(one_ec + 1 + SYS_N/8,  c, SYND_BYTES);

    // Derive 32-byte shared key via SHAKE256
    crypto_hash_32b(key, one_ec, sizeof(one_ec));

    return 0;
}

