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

extern "C" {

void masked_softmax_float32(float attention_score[32][64],
                            int tile_row_start[1], float attn_weights[32][64]) {
  // Define constants for tile dimensions and vectorization
  constexpr int TILE_ROWS = 32;
  constexpr int SEQ_COLS = 64;
  constexpr int VEC_SIZE = 32;

  // Define negative infinity constant
  const float neg_inf = -std::numeric_limits<float>::infinity();
  aie::vector<float, VEC_SIZE> log2e_vec =
      aie::broadcast<float, VEC_SIZE>(log2e);
  aie::vector<bfloat16, VEC_SIZE> scale_vec =
      aie::broadcast<bfloat16, VEC_SIZE>(1.0);
  // Loop over each row in the tile
  for (int r = 0; r < TILE_ROWS; ++r) {
    // Calculate global row index for causal masking
    int global_row_idx = tile_row_start[0] + r;

    // Pointers for current row's input and output
    float *__restrict current_attention_score_row_ptr = &attention_score[r][0];
    float *__restrict current_attn_weights_row_ptr = &attn_weights[r][0];

    aie::vector<float, VEC_SIZE> scores_v0 =
        aie::load_v<VEC_SIZE>(current_attention_score_row_ptr);
    aie::vector<float, VEC_SIZE> scores_v1 =
        aie::load_v<VEC_SIZE>(current_attention_score_row_ptr + VEC_SIZE);

    scores_v0 = aie::mul(scores_v0, log2e_vec);
    aie::store_v(current_attention_score_row_ptr, scores_v0);

    scores_v1 = aie::mul(scores_v1, log2e_vec);
    aie::store_v(current_attention_score_row_ptr + VEC_SIZE, scores_v1);

    for (int k = 0; k < SEQ_COLS; ++k) {
      if (k > global_row_idx) {
        attention_score[r][k] = neg_inf;
      }
    }
    scores_v0 = aie::load_v<VEC_SIZE>(current_attention_score_row_ptr);
    scores_v1 =
        aie::load_v<VEC_SIZE>(current_attention_score_row_ptr + VEC_SIZE);

    // --- Find Max Value for Numerical Stability (LogSumExp trick) ---
    float row_max = aie::reduce_max(scores_v0);
    row_max = std::max(row_max, aie::reduce_max(scores_v1));
    scores_v0 = aie::add(scores_v0, -row_max);
    scores_v1 = aie::add(scores_v1, -row_max);
    // --- Compute exp(x - max) using scalar approximation ---
    float sum_exp = 0.0f;

    auto exp_vec0 = aie::exp2<bfloat16>(scores_v0); // ! require XDNA2
    aie::accum<accfloat, VEC_SIZE> exp_out0 = aie::mul(exp_vec0, scale_vec);
    auto attn_weight0 = exp_out0.to_vector<float>();
    aie::store_v(current_attn_weights_row_ptr + VEC_SIZE, attn_weight0);
    sum_exp += aie::reduce_add(attn_weight0);

    auto exp_vec1 = aie::exp2<bfloat16>(scores_v1); // ! require XDNA2
    aie::accum<accfloat, VEC_SIZE> exp_out1 = aie::mul(exp_vec1, scale_vec);
    auto attn_weight1 = exp_out1.to_vector<float>();
    aie::store_v(current_attn_weights_row_ptr + VEC_SIZE, attn_weight1);
    sum_exp += aie::reduce_add(attn_weight1);

    float scale = 1.0f / sum_exp;
    aie::vector<float, VEC_SIZE> weight_v0 =
        aie::load_v<VEC_SIZE>(current_attn_weights_row_ptr);
    aie::vector<float, VEC_SIZE> weight_v1 =
        aie::load_v<VEC_SIZE>(current_attn_weights_row_ptr + VEC_SIZE);
    weight_v0 = aie::mul(weight_v0, scale);
    weight_v1 = aie::mul(weight_v1, scale);
    aie::store_v(current_attn_weights_row_ptr, weight_v0);
    aie::store_v(current_attn_weights_row_ptr + VEC_SIZE, weight_v1);
  }
}

} // extern "C"