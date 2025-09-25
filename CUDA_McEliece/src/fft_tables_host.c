#include <stdint.h>
#include "../include/params.h"

// These four arrays live on the host. The .data files should be in your include/ dir.

const uint64_t H_scalars[5][GFBITS] = {
#include "scalars.data"
};

const uint64_t H_consts[63][GFBITS] = {
#include "consts.data"
};

const uint64_t H_powers[64][GFBITS] = {
#include "powers.data"
};

const uint64_t H_scalars2x[5][2][GFBITS] = {
#include "scalars_2x.data"
};
