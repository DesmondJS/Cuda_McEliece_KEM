/*
  This file is for transpose of the Gao-Mateer FFT
*/

#ifndef FFT_TR_H
#define FFT_TR_H
#include "namespace.h"
#define fft_tr CRYPTO_NAMESPACE(fft_tr)
#include "params.h"
#include "vec.h"

#ifdef __cplusplus
extern "C" {
#endif

void fft_tr(vec [][GFBITS], vec [][ GFBITS ]);

#ifdef __cplusplus
}
#endif


#endif

