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

bfloat16 get_exp(bfloat16 input_x) {
  float x = (float)input_x;
  // Improved exp approximation
  // log2(e) = 1.442695040888963
  int32_t ix = (int32_t)(x * 1.442695040888963f);
  float fx = x * 1.442695040888963f - ix;
  // Compute 2^ix using bit manipulation
  ix = (ix + 127) << 23;
  float pow2_ix;
  memcpy(&pow2_ix, &ix, sizeof(float));
  // Improved approximation for 2^fx with correction term
  // ln(2) = 0.6931471805599453
  float pow2_fx =
      1.0f + 0.6931471805599453f * fx + 0.2401598148889220f * fx * fx;
  float result = pow2_ix * pow2_fx;
  return (bfloat16)result;
}


extern "C" {

void softmax_bfloat16(bfloat16 attention_score[2][1024],
                      bfloat16 attention_weight[2][1024]) {
  // Define constants for tile dimensions and vectorization
  constexpr int TILE_ROWS = 2;
  constexpr int vec_factor = 256 / (sizeof(bfloat16) * 8);
  const int F = 1024 / vec_factor;
  // Define negative infinity constant
  const bfloat16 neg_inf = bfloat16(-std::numeric_limits<float>::infinity());
  // Loop over each row in the tile
  for (int r = 0; r < TILE_ROWS; ++r) {
    bfloat16 *score_row_ptr = &attention_score[r][0];
    // row max
    bfloat16 row_max = neg_inf;
    for (int i = 0; i < F; i++) {
      aie::vector<bfloat16, vec_factor> scores =
          aie::load_v<vec_factor>(score_row_ptr);
      row_max = std::max(row_max, aie::reduce_max(scores));
      score_row_ptr += vec_factor;
    }
    // exp logit
    bfloat16 exp_sum = bfloat16(0.0f);
    for (int i = 0; i < 1024; i++) {
      bfloat16 exp_result = get_exp(attention_score[r][i] - row_max);
      attention_weight[r][i] = exp_result;
      exp_sum += exp_result;
    }
    bfloat16 scale = 1.0f / exp_sum;
    bfloat16 *weight_row_ptr = &attention_weight[r][0];
    for (int i = 0; i < F; i++) {
      aie::vector<bfloat16, vec_factor> weight =
          aie::load_v<vec_factor>(weight_row_ptr);
      weight = aie::mul(weight, scale);
      aie::store_v(weight_row_ptr, weight);
      weight_row_ptr += vec_factor;
    }
  }
}

} // extern "C"