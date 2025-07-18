/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define NOCPP


template <typename T_in, typename T_out, const int SEQ_LEN, const int HIDDEN>
void customized_exp(T_in *input_tensor, T_out *output_tensor) {
  constexpr int vec_factor = 16;
  using vec_t = aie::vector<T_in, vec_factor>;
  event0();
  
  event1();
}

extern "C" {

void my_exp(float A_in[1][64], float A_out[1][64]) {
  customized_exp<float, float, 1, 64>(&A_in[0][0], &A_out[0][0]);
}

} // extern "C"
