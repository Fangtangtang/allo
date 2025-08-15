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
#define log2e 1.44269504089

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

template <int L>
void init_softmax(bfloat16 *__restrict max_logit,
                  bfloat16 *__restrict sum_exp) {
  // max_logit = np.full((L, 1), -np.inf)
  // sum_exp = np.zeros((L, 1))
  constexpr int vec_factor =
      256 / (sizeof(bfloat16) * 8); // one 256 bit store unit
  static_assert(L % vec_factor == 0);
  const bfloat16 neg_inf = bfloat16(-std::numeric_limits<float>::infinity());
  const aie::vector<bfloat16, vec_factor> neg_infs =
      aie::broadcast<bfloat16, vec_factor>(neg_inf);
  const aie::vector<bfloat16, vec_factor> zeros =
      aie::zeros<bfloat16, vec_factor>();
  for (int iter = 0; iter < L; iter += vec_factor) {
    aie::store_v(max_logit, neg_infs);
    max_logit += vec_factor;
    aie::store_v(sum_exp, zeros);
    sum_exp += vec_factor;
  }
}

extern "C" {

void init_softmax(bfloat16 max_logit[32], bfloat16 sum_exp[32]) {
  init_softmax<32>(max_logit, sum_exp);
}

void online_softmax(bfloat16 attention_score[32][32],
                    bfloat16 prev_max_logit[32], bfloat16 prev_sum_exp[32],
                    bfloat16 attention_weight[32][32],
                    bfloat16 new_max_logit[32], bfloat16 new_sum_exp[32]) {
  constexpr int vec_factor = 256 / (sizeof(bfloat16) * 8);
  const int F = 32 / vec_factor;
  for (int r = 0; r < 32; ++r) {
    bfloat16 *score_row_ptr = &attention_score[r][0];
    // row max
    bfloat16 row_max = prev_max_logit[r];
    for (int i = 0; i < F; i++) {
      aie::vector<bfloat16, vec_factor> scores =
          aie::load_v<vec_factor>(score_row_ptr);
      row_max = std::max(row_max, aie::reduce_max(scores));
      score_row_ptr += vec_factor;
    }
    new_max_logit[r] = row_max;
    // exp logit
    bfloat16 exp_sum = prev_sum_exp[r];
    for (int i = 0; i < 32; i++) {
      bfloat16 exp_result = get_exp(attention_score[r][i] - row_max);
      attention_weight[r][i] = exp_result;
      exp_sum += exp_result;
    }
    new_sum_exp[r] = exp_sum;
  }
}

void online_softmax2(bfloat16 attention_score[32][32],
                     bfloat16 prev_max_logit[32], bfloat16 prev_sum_exp[32],
                     bfloat16 attention_weight[32][32],
                     bfloat16 new_max_logit[32], bfloat16 new_sum_exp[32]) {
  constexpr int vec_factor = 256 / (sizeof(bfloat16) * 8);
  const int F = 32 / vec_factor;
  for (int r = 0; r < 32; ++r) {
    bfloat16 *score_row_ptr = &attention_score[r][0];
    // row max
    bfloat16 row_max = prev_max_logit[r];
    for (int i = 0; i < F; i++) {
      aie::vector<bfloat16, vec_factor> scores =
          aie::load_v<vec_factor>(score_row_ptr);
      row_max = std::max(row_max, aie::reduce_max(scores));
      score_row_ptr += vec_factor;
    }
    new_max_logit[r] = row_max;
    // exp logit
    bfloat16 exp_sum = prev_sum_exp[r];
    aie::vector<bfloat16, vec_factor> log2e_vec =
        aie::broadcast<bfloat16, vec_factor>(log2e);
    for (int i = 0; i < F; i++) {
      bfloat16 *score_row_ptr = &attention_score[r][0];
      bfloat16 *weight_row_ptr = &attention_weight[r][0];
      aie::vector<bfloat16, vec_factor> scores =
          aie::load_v<vec_factor>(score_row_ptr);
      aie::accum<accfloat, vec_factor> exp_in =
          aie::mul(aie::sub(scores, row_max), log2e_vec);
      aie::vector<bfloat16, vec_factor> exp_val =
          aie::exp2<bfloat16>(exp_in.to_vector<float>());
      aie::store_v(weight_row_ptr, exp_val);
      exp_sum += aie::reduce_add(exp_val);
      score_row_ptr += vec_factor;
      weight_row_ptr += vec_factor;
    }
    new_sum_exp[r] = exp_sum;
  }
}

} // extern "C"