/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
 
#include <aie_api/aie.hpp>
#include <stdint.h>

#define VEC_LEN 32
#define log2e 1.44269504089

using namespace aie;

template <const int N>
void exp_bf16_func(bfloat16 *restrict in, bfloat16 *restrict out) {
  auto it_exp_in = aie::cbegin_vector<VEC_LEN>((bfloat16 *)in);
  auto it_exp_out = aie::begin_vector<VEC_LEN>((bfloat16 *)out);

  const int elem_iters = N / VEC_LEN;

  // Calculate the e^(x) function as 2^(log2e * x)
  aie::vector<bfloat16, VEC_LEN> input_bf16;
  aie::accum<accfloat, VEC_LEN> exp_in;
  aie::vector<bfloat16, VEC_LEN> exp_val;
  aie::vector<bfloat16, VEC_LEN> log2e_vec =
      aie::broadcast<bfloat16, VEC_LEN>(log2e);

  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_exp_in++;
    exp_in = aie::mul(input_bf16, log2e_vec);
    exp_val = aie::exp2<bfloat16>(exp_in.to_vector<float>());
    *it_exp_out++ = exp_val;
  }
}

template <const int N>
void softmax_simple_bf16(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector) {
  event0();
  // VJUNG: We do 3 passes on the vector:
  // 1. Find the max value scaled by log2e in the vector
  // 2. Calculate the exponentials of the scaled values minus the maximum
  // 3. Calculate the softmax by dividing each exponential by the sum of all
  // exponentials Note: The multiplication by log2e is very sensitive, casting
  // it to bf16 before exponentiation leads to wrong output.
  auto it_log_in =
      aie::cbegin_restrict_vector<VEC_LEN>((bfloat16 *)input_vector);
  auto it_log_out =
      aie::begin_restrict_vector<VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_in =
      aie::cbegin_restrict_vector<VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_out =
      aie::begin_restrict_vector<VEC_LEN>((bfloat16 *)output_vector);
  auto it_scale =
      aie::cbegin_restrict_vector<VEC_LEN>((bfloat16 *)output_vector);
  auto it_soft_out =
      aie::begin_restrict_vector<VEC_LEN>((bfloat16 *)output_vector);

  aie::vector<bfloat16, VEC_LEN> in_elems, exp_val, input_bf16, log2e_vec,
      max_val_vec;
  aie::accum<accfloat, VEC_LEN> out_vals, exp_val_accum, scaled_accum,
      exp_in_accum;

  float max_val = 0;
  float accum_exp_val = 0;
  float running_max = 0;
  bfloat16 col_sum_inv;
  const int elem_iters = N / VEC_LEN;

  exp_val_accum = aie::zeros<accfloat, VEC_LEN>();

  log2e_vec = aie::broadcast<bfloat16, VEC_LEN>((bfloat16)log2e);

  // First pass - Optimized: element-wise max + single final reduce_max
  // Use vector max accumulation, then reduce once at the end
  aie::vector<bfloat16, VEC_LEN> max_accum_vec =
      aie::broadcast<bfloat16, VEC_LEN>((bfloat16)-32768.0f);
  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_log_in++;
    scaled_accum = aie::mul(input_bf16, log2e_vec);
    max_accum_vec = aie::max(max_accum_vec, scaled_accum.to_vector<bfloat16>());
  }
  max_val = aie::reduce_max(max_accum_vec);
  max_val_vec = aie::broadcast<bfloat16, VEC_LEN>(max_val);

  // Second pass
  for (int i = 0; i < elem_iters; i++) {

    input_bf16 = *it_exp_in++;

    scaled_accum = aie::mul(input_bf16, log2e_vec);
    exp_in_accum = aie::sub(scaled_accum, max_val_vec);
    exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
    exp_val_accum = add(exp_val_accum, exp_val);

    *it_exp_out++ = exp_val;
  }

  // Final reduction after loop
  aie::vector<float, VEC_LEN> reduce = exp_val_accum.to_vector<float>();
  accum_exp_val = aie::reduce_add(reduce);
  col_sum_inv = (bfloat16)aie::inv(accum_exp_val);

  for (int c = 0; c < elem_iters; c++) {
    in_elems = *it_scale++;
    out_vals = aie::mul(in_elems, col_sum_inv);
    *it_soft_out++ = out_vals.to_vector<bfloat16>();
  }

  event1();
  return;
}

template <int L>
void init_softmax(bfloat16 *__restrict max_logit,
                  bfloat16 *__restrict sum_exp) {
  // max_logit = np.full((L, 1), -np.inf)
  // sum_exp = np.zeros((L, 1))
  constexpr int vec_factor = 512 / (sizeof(bfloat16) * 8); // one 256 bit store unit
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

void exp_bf16(bfloat16 a_in[1024], bfloat16 c_out[1024]) {
  exp_bf16_func<1024>(a_in, c_out);
}

void vector_softmax_bf16(bfloat16 a_in[1024], bfloat16 c_out[1024]) {
  softmax_simple_bf16<1024>(a_in, c_out);
}

void init_softmax(bfloat16 max_logit[64], bfloat16 sum_exp[64]) {
  init_softmax<64>(max_logit, sum_exp);
}

void online_softmax(bfloat16 attention_score[64][64],
                    bfloat16 prev_max_logit[64], bfloat16 prev_sum_exp[64],
                    bfloat16 attention_weight[64][64], bfloat16 scale_exp[64]) {
  // Note: prev_max_logit/new_max_logit and prev_sum_exp/new_sum_exp alias the
  // same buffers at the call site, so each `prev_*[r]` is read before the
  // matching `new_*[r]` is written, and none of them are marked restrict.
  constexpr int N = 64; // one row == one vector
  const bfloat16 LS = bfloat16(0.125f * log2e); // == 0.1806640625
  alignas(aie::vector_decl_align) float row_sum[N]; // per-row sum(exp), kept in fp for precision

  for (int r = 0; r < N; r++) {
    aie::vector<bfloat16, N> scores = aie::load_v<N>(&attention_score[r][0]);
    aie::accum<accfloat, N> logits = aie::mul(scores, LS); // log2-domain logits
    bfloat16 local_max = aie::reduce_max(logits.template to_vector<bfloat16>());
    bfloat16 prev_max = prev_max_logit[r];
    bfloat16 row_max = (prev_max > local_max) ? prev_max : local_max;

    // Stash (prev_max - row_max) into scale_exp; exp2'd in the tail below.
    scale_exp[r] = bfloat16(prev_max - row_max);

    aie::accum<accfloat, N> shifted =
        aie::sub(logits, aie::broadcast<bfloat16, N>(row_max));
    aie::vector<bfloat16, N> exp_val =
        aie::exp2<bfloat16>(shifted.template to_vector<float>());
    aie::store_v(&attention_weight[r][0], exp_val);

    aie::accum<accfloat, N> sum_acc;
    sum_acc.from_vector(exp_val, 0);
    row_sum[r] = aie::reduce_add(sum_acc.template to_vector<float>());

    prev_max_logit[r] = row_max; // safe: prev_max already read (alias)
  }

  // Tail 1: scale_exp = exp2(prev_max - new_max) (already in log2 units).
  {
    aie::vector<bfloat16, N> d = aie::load_v<N>(scale_exp);
    aie::accum<accfloat, N> da;
    da.from_vector(d, 0);
    aie::store_v(scale_exp, aie::exp2<bfloat16>(da.template to_vector<float>()));
  }

  // Tail 2: new_sum_exp = prev_sum_exp * scale_exp + row_sum (vectorized).
  {
    aie::vector<bfloat16, N> ps = aie::load_v<N>(prev_sum_exp); // before write
    aie::vector<bfloat16, N> se = aie::load_v<N>(scale_exp);
    aie::vector<float, N> rs = aie::load_v<N>(row_sum);
    aie::accum<accfloat, N> acc = aie::mul(ps, se);
    acc = aie::add(acc, rs);
    aie::store_v(prev_sum_exp, acc.template to_vector<bfloat16>());
  }
}

} // extern "C"
