/*
  This file is for the inversion-free Berlekamp-Massey algorithm
  see https://ieeexplore.ieee.org/document/87857
*/

#ifndef BM_H
#define BM_H
#include "namespace.h"
#include <stdint.h>
#include "params.h"
#include "vec.h"

#define bm CRYPTO_NAMESPACE(bm)


#ifdef __cplusplus
extern "C" {
#endif

void bm(vec *, vec [][ GFBITS ]);

#ifdef __cplusplus
}
#endif

#endif

