/* This file uses SHAKE256 implemented in the Keccak Code Package */

// #ifdef __cplusplus
// extern "C" {
// #endif
// #include "../libkeccak.a.headers/SimpleFIPS202.h"
// #ifdef __cplusplus
// }
// #endif

// #define crypto_hash_32b(out,in,inlen) \
//   SHAKE256(out,32,in,inlen)

// #define shake(out,outlen,in,inlen) \
//   SHAKE256(out,outlen,in,inlen)

#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Declaration only; implemented in src/shake_shim.c */
int SHAKE256(unsigned char *output, size_t outputByteLen,
             const unsigned char *input,  size_t inputByteLen);

#ifdef __cplusplus
}
#endif

/* Keep your convenience macros exactly as before */
#define crypto_hash_32b(out,in,inlen) SHAKE256(out, 32, in, inlen)
#define shake(out,outlen,in,inlen)    SHAKE256(out, outlen, in, inlen)