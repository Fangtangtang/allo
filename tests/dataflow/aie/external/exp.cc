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
  const int F = HIDDEN / vec_factor;
  using vec_t = aie::vector<T_in, vec_factor>;
  event0();
  for (int iter = 0; iter < SEQ_LEN; iter++) {
    T_in *__restrict input_ptr = input_tensor;
    T_out *__restrict output_ptr = output_tensor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
    // TODO
      aie::store_v(output_ptr, input_vec);
      output_ptr += vec_factor;
    }
  }
  event1();
}

extern "C" {

void my_exp(bfloat16 A_in[1][64], bfloat16 A_out[1][64]) {
  customized_exp<bfloat16, bfloat16, 1, 64>(&A_in[0][0], &A_out[0][0]);
}

} // extern "C"
