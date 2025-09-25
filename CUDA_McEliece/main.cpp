// Include C++ header files.
#include <iostream>
#include <cstring>
#include <stdio.h>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"
#include "include/cuda_decrypt.cuh"
#include "include/crypto_kem.h"
#include "include/crypto_kem_mceliece348864.h"
#include "include/keys.cuh"
#include "include/params.h"
#include "include/rng.h"

unsigned char entropy_input[48];
unsigned char seed[KATNUM][48];
unsigned char dummy[32];

int main() {

    //Act as seed for random number generator
    for (int i=0; i<48; i++)
        entropy_input[i] = i;
    randombytes_init(entropy_input, NULL, 256);

    for (int i=0; i<KATNUM; i++){
        randombytes(seed[i], 48);
        randombytes_init(seed[i], NULL, 256);
    }

    //We dont run crypto_kem_keypair so have to push the randombytes stream
    randombytes(dummy, 32);

    // Local buffers for outputs (don’t overwrite any expected vectors)
    unsigned char ct_local[crypto_kem_CIPHERTEXTBYTES];
    unsigned char ss_enc  [crypto_kem_BYTES];
    unsigned char ss_dec  [crypto_kem_BYTES];

    //Encapsulate with public key -> ct, ss
    if (crypto_kem_enc_new(ct_local, ss_enc, pk) != 0) {
        printf("crypto_kem_enc_new failed\n");
        return 1;
    }

    //Decapsulate with secret key -> ss1
    if (crypto_kem_dec(ss_dec, ct_local, sk) != 0) {
        printf("crypto_kem_dec failed (nonzero return)\n");
        return 1;
    }

    //Compare shared secrets
    if (std::memcmp(ss_enc, ss_dec, crypto_kem_BYTES) != 0) {
        printf("Mismatch: ss (enc) != ss1 (dec)\n");
        return 1;
    }

    printf("Success: ss (enc) == ss1 (dec)\n");
    return 0;
}