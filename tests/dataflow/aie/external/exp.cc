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
#define log2e 1.4453125

template <typename T_in, typename T_out, const int SEQ_LEN, const int HIDDEN>
void customized_exp(T_in *input_tensor, T_out *output_tensor) {
  constexpr int vec_factor = 16;
  const int F = HIDDEN / vec_factor;
  using vec_t = aie::vector<T_in, vec_factor>;
  event0();
  aie::vector<bfloat16, vec_factor> log2e_vec =
      aie::broadcast<bfloat16, vec_factor>(log2e);
  aie::vector<bfloat16, vec_factor> scale_vec =
      aie::broadcast<bfloat16, vec_factor>(1.0);
  for (int iter = 0; iter < SEQ_LEN; iter++) {
    T_in *__restrict input_ptr = input_tensor;
    T_out *__restrict output_ptr = output_tensor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      input_vec = aie::mul(input_vec, log2e_vec);
      auto result_vec = aie::exp2<bfloat16>(input_vec); // ! require XDNA2
      aie::accum<accfloat, vec_factor> exp_out = aie::mul(result_vec, scale_vec);
      aie::store_v(output_ptr, (exp_out.to_vector<float>()));
      output_ptr += vec_factor;
    }
  }
  event1();
}

extern "C" {

void my_exp(float A_in[1][64], float A_out[1][64]) {
  customized_exp<float, float, 1, 64>(&A_in[0][0], &A_out[0][0]);
}

} // extern "C"
