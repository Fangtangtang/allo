// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Stubbed matmul microkernel for DRAM <-> NPU bandwidth micro-benchmark.
// Provides the same template helper names and `matmul_*_c_func` macros as
// allo/library/aie/kernels/mm.cc so that GEMM(...) builds without changes.
// Every helper writes zeros to the output buffer instead of accumulating
// products, which strips MAC time from the measurement while leaving every
// shim/mem DMA descriptor identical to a real GEMM run.

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>
#include "aie2/zero.cc"

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using bfloat16 = __bf16;

template <typename T_out, int M, int N>
static inline void stub_zero(T_out *__restrict c) {
  constexpr int r = 256 / (sizeof(T_out) * 8);
  if constexpr ((M * N) % r == 0) {
    zero_vectorized<T_out, M, N>(c);
  } else {
    for (int i = 0; i < M * N; i++) {
      c[i] = (T_out)0;
    }
  }
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
static inline void matmul_scalar(T_in *a, T_in *b, T_out *c) {
  (void)a;
  (void)b;
  stub_zero<T_out, rowA, colB>(c);
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_2x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {
  (void)pA;
  (void)pB;
  stub_zero<T_out, rowA * r, colB * t>(pC);
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_4x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {
  (void)pA;
  (void)pB;
  stub_zero<T_out, rowA * r, colB * t>(pC);
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_4x4(const T_in *__restrict pA,
                                         const T_in *__restrict pB,
                                         T_out *__restrict pC) {
  (void)pA;
  (void)pB;
  stub_zero<T_out, rowA * r, colB * t>(pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x4x4_i16_i16(const int16 *__restrict pA,
                                                   const int16 *__restrict pB,
                                                   int16 *__restrict pC) {
  (void)pA;
  (void)pB;
  stub_zero<int16, m, n>(pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x4x4_i16_i32(const int16 *__restrict pA,
                                                   const int16 *__restrict pB,
                                                   int32 *__restrict pC) {
  (void)pA;
  (void)pB;
  stub_zero<int32, m, n>(pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_4x8x4_bf16_bf16(const bfloat16 *__restrict pA,
                                  const bfloat16 *__restrict pB,
                                  bfloat16 *__restrict pC) {
  (void)pA;
  (void)pB;
  stub_zero<bfloat16, m, n>(pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_4x8x4_bf16_f32(const bfloat16 *__restrict pA,
                                 const bfloat16 *__restrict pB,
                                 float *__restrict pC) {
  (void)pA;
  (void)pB;
  stub_zero<float, m, n>(pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x8x8_i8_i8(const int8 *__restrict pA,
                                                 const int8 *__restrict pB,
                                                 int8 *__restrict pC) {
  (void)pA;
  (void)pB;
  stub_zero<int8, m, n>(pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x16x8_i4_i8_packedB(
    const int8 *__restrict pA, const int8 *__restrict pB,
    int8 *__restrict pC) {
  (void)pA;
  (void)pB;
  stub_zero<int8, m, n>(pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x16x8_i4_i8(const int8 *__restrict pA,
                                                  const int8 *__restrict pB,
                                                  int8 *__restrict pC) {
  (void)pA;
  (void)pB;
  stub_zero<int8, m, n>(pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x8x8_i8_i16(const int8 *__restrict pA,
                                                  const int8 *__restrict pB,
                                                  int16 *__restrict pC) {
  (void)pA;
  (void)pB;
  stub_zero<int16, m, n>(pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x8x8_i8_i32(const int8 *__restrict pA,
                                                  const int8 *__restrict pB,
                                                  int32 *__restrict pC) {
  (void)pA;
  (void)pB;
  stub_zero<int32, m, n>(pC);
}

#define CAT(a, b) a##b
#define CAT3(a, b, c) a##b##c
#define CAT5(a, b, c, d, e) a##b##c##d##e
#define EXPAND_AND_CAT(a, b) CAT(a, b)
#define EXPAND_AND_CAT3(a, b, c) CAT3(a, b, c)
#define EXPAND_AND_CAT5(a, b, c, d, e) CAT5(a, b, c, d, e)

#define GEN_VECTOR_FUNC_NAME(type_in, type_out, m, k, n)                       \
  EXPAND_AND_CAT5(matmul_##type_in##_##type_out##_, m, x, k,                   \
                  EXPAND_AND_CAT(x, n))

#define GEN_SCALAR_FUNC_NAME(type_in, type_out, m, k, n)                       \
  EXPAND_AND_CAT5(matmul_scalar_##type_in##_##type_out##_, m, x, k,            \
                  EXPAND_AND_CAT(x, n))

#define matmul_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, r, s, t, DIM_M_, DIM_K_,       \
                                 DIM_N_)                                       \
  void GEN_VECTOR_FUNC_NAME(mlir_type_in, mlir_type_out, DIM_M_, DIM_K_,       \
                            DIM_N_)(ctype_in * a_in, ctype_in * b_in,          \
                                    ctype_out * c_out) {                       \
    matmul_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out<      \
        DIM_M_, DIM_K_, DIM_N_>(a_in, b_in, c_out);                            \
  }

#define matmul_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, \
                             r, s, t, DIM_M_, DIM_K_, DIM_N_)                  \
  void GEN_SCALAR_FUNC_NAME(mlir_type_in, mlir_type_out, DIM_M_, DIM_K_,       \
                            DIM_N_)(ctype_in * a_in, ctype_in * b_in,          \
                                    ctype_out * c_out) {                       \
    matmul_scalar<ctype_in, ctype_out, DIM_M_, DIM_K_, DIM_N_>(a_in, b_in,     \
                                                               c_out);         \
  }
