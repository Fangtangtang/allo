//===- mm.cc ----------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "zero.cc"

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using bfloat16 = __bf16;

static inline int8 sign_extend_nibble(uint8_t nibble) {
  return static_cast<int8>(static_cast<int8>(nibble << 4) >> 4);
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
static inline void matmul_scalar(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      c[row * colB + col] += running_sum;
    }
  }
  event1();
}

// Packed i4 variant: B stored with two 4-bit values per byte along N; unpack to
// two int4 vectors per j-pair and compute two output columns per iteration.
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x32x8_i4_i8_packed(
    const int8 *__restrict pA, const int8 *__restrict pB, int8 *__restrict pC) {

  constexpr int r = 4;
  constexpr int s = 16;
  constexpr int t = 8;

  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  using MMUL = aie::mmul<r, s, t, int8, int4, accauto>;

  event0();

  for (unsigned z = 0; z < (m / r); z += 4)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      int8 *__restrict pC1 = pC + (z * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC2 = pC + ((z + 1) * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC3 = pC + ((z + 2) * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC4 = pC + ((z + 3) * (n / t) + 0) * MMUL::size_C;

      for (unsigned j = 0; j < (n / t); j += 2) {
        const int8 *__restrict pA1 = pA + (z * (k / s) + 0) * MMUL::size_A;
        const int8 *__restrict pA2 =
            pA + ((z + 1) * (k / s) + 0) * MMUL::size_A;
        const int8 *__restrict pA3 =
            pA + ((z + 2) * (k / s) + 0) * MMUL::size_A;
        const int8 *__restrict pA4 =
            pA + ((z + 3) * (k / s) + 0) * MMUL::size_A;

        unsigned jpair = j / 2;
        const uint8_t *__restrict bBytes =
            reinterpret_cast<const uint8_t *>(pB);
        const unsigned bytesPerRow = (n / 2);

        aie::vector<int8, MMUL::size_A> A01 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A11 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A21 = aie::load_v<MMUL::size_A>(pA3);
        pA3 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A31 = aie::load_v<MMUL::size_A>(pA4);
        pA4 += MMUL::size_A;

        alignas(16) int8 tmpB_blk0_u8[MMUL::size_B];
        alignas(16) int8 tmpB_blk1_u8[MMUL::size_B];
        // Unpack first K-chunk (i = 0): s rows, 2 contiguous t-column blocks (j
        // and j+1)
        {
          unsigned rowBase = 0;
          for (unsigned s_lane = 0; s_lane < s; ++s_lane)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              const uint8_t *rowPtr = bBytes + (rowBase + s_lane) * bytesPerRow;
              const unsigned outBase = s_lane * t;

              const unsigned base0 = (j * t) >> 1;       // j*4 bytes
              const unsigned base1 = ((j + 1) * t) >> 1; // (j+1)*4 bytes

              uint8_t b0 = rowPtr[base0 + 0];
              uint8_t b1 = rowPtr[base0 + 1];
              uint8_t b2 = rowPtr[base0 + 2];
              uint8_t b3 = rowPtr[base0 + 3];

              tmpB_blk0_u8[outBase + 0] = sign_extend_nibble(b0 & 0x0F);
              tmpB_blk0_u8[outBase + 1] = sign_extend_nibble(b0 >> 4);
              tmpB_blk0_u8[outBase + 2] = sign_extend_nibble(b1 & 0x0F);
              tmpB_blk0_u8[outBase + 3] = sign_extend_nibble(b1 >> 4);
              tmpB_blk0_u8[outBase + 4] = sign_extend_nibble(b2 & 0x0F);
              tmpB_blk0_u8[outBase + 5] = sign_extend_nibble(b2 >> 4);
              tmpB_blk0_u8[outBase + 6] = sign_extend_nibble(b3 & 0x0F);
              tmpB_blk0_u8[outBase + 7] = sign_extend_nibble(b3 >> 4);

              uint8_t b4 = rowPtr[base1 + 0];
              uint8_t b5 = rowPtr[base1 + 1];
              uint8_t b6 = rowPtr[base1 + 2];
              uint8_t b7 = rowPtr[base1 + 3];

              tmpB_blk1_u8[outBase + 0] = sign_extend_nibble(b4 & 0x0F);
              tmpB_blk1_u8[outBase + 1] = sign_extend_nibble(b4 >> 4);
              tmpB_blk1_u8[outBase + 2] = sign_extend_nibble(b5 & 0x0F);
              tmpB_blk1_u8[outBase + 3] = sign_extend_nibble(b5 >> 4);
              tmpB_blk1_u8[outBase + 4] = sign_extend_nibble(b6 & 0x0F);
              tmpB_blk1_u8[outBase + 5] = sign_extend_nibble(b6 >> 4);
              tmpB_blk1_u8[outBase + 6] = sign_extend_nibble(b7 & 0x0F);
              tmpB_blk1_u8[outBase + 7] = sign_extend_nibble(b7 >> 4);
            }
        }
        aie::vector<int8, MMUL::size_B> B_blk0_u8 =
            aie::load_v<MMUL::size_B>(tmpB_blk0_u8);
        aie::vector<int8, MMUL::size_B> B_blk1_u8 =
            aie::load_v<MMUL::size_B>(tmpB_blk1_u8);
        aie::vector<int4, MMUL::size_B> B01 = B_blk0_u8.pack();
        aie::vector<int4, MMUL::size_B> B11 = B_blk1_u8.pack();

        aie::vector<int8, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pC1);
        aie::vector<int8, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
        aie::vector<int8, MMUL::size_C> acc_C10 =
            aie::load_v<MMUL::size_C>(pC2);
        aie::vector<int8, MMUL::size_C> acc_C11 =
            aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
        aie::vector<int8, MMUL::size_C> acc_C20 =
            aie::load_v<MMUL::size_C>(pC3);
        aie::vector<int8, MMUL::size_C> acc_C21 =
            aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
        aie::vector<int8, MMUL::size_C> acc_C30 =
            aie::load_v<MMUL::size_C>(pC4);
        aie::vector<int8, MMUL::size_C> acc_C31 =
            aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);
        MMUL C20(acc_C20);
        MMUL C21(acc_C21);
        MMUL C30(acc_C30);
        MMUL C31(acc_C31);

        C00.mac(A01, B01);
        C01.mac(A01, B11);
        C10.mac(A11, B01);
        C11.mac(A11, B11);
        C20.mac(A21, B01);
        C21.mac(A21, B11);
        C30.mac(A31, B01);
        C31.mac(A31, B11);

        for (unsigned i = 1; i < (k / s); i += 1) {
          A01 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          A11 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          A21 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += MMUL::size_A;
          A31 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += MMUL::size_A;

          // Unpack K-chunk i: rows [i*s .. i*s + s-1]
          unsigned rowBase = i * s;
          for (unsigned s_lane = 0; s_lane < s; ++s_lane)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              const uint8_t *rowPtr = bBytes + (rowBase + s_lane) * bytesPerRow;
              const unsigned outBase = s_lane * t;

              const unsigned base0 = (j * t) >> 1;
              const unsigned base1 = ((j + 1) * t) >> 1;

              uint8_t b0 = rowPtr[base0 + 0];
              uint8_t b1 = rowPtr[base0 + 1];
              uint8_t b2 = rowPtr[base0 + 2];
              uint8_t b3 = rowPtr[base0 + 3];

              tmpB_blk0_u8[outBase + 0] = sign_extend_nibble(b0 & 0x0F);
              tmpB_blk0_u8[outBase + 1] = sign_extend_nibble(b0 >> 4);
              tmpB_blk0_u8[outBase + 2] = sign_extend_nibble(b1 & 0x0F);
              tmpB_blk0_u8[outBase + 3] = sign_extend_nibble(b1 >> 4);
              tmpB_blk0_u8[outBase + 4] = sign_extend_nibble(b2 & 0x0F);
              tmpB_blk0_u8[outBase + 5] = sign_extend_nibble(b2 >> 4);
              tmpB_blk0_u8[outBase + 6] = sign_extend_nibble(b3 & 0x0F);
              tmpB_blk0_u8[outBase + 7] = sign_extend_nibble(b3 >> 4);

              uint8_t b4 = rowPtr[base1 + 0];
              uint8_t b5 = rowPtr[base1 + 1];
              uint8_t b6 = rowPtr[base1 + 2];
              uint8_t b7 = rowPtr[base1 + 3];

              tmpB_blk1_u8[outBase + 0] = sign_extend_nibble(b4 & 0x0F);
              tmpB_blk1_u8[outBase + 1] = sign_extend_nibble(b4 >> 4);
              tmpB_blk1_u8[outBase + 2] = sign_extend_nibble(b5 & 0x0F);
              tmpB_blk1_u8[outBase + 3] = sign_extend_nibble(b5 >> 4);
              tmpB_blk1_u8[outBase + 4] = sign_extend_nibble(b6 & 0x0F);
              tmpB_blk1_u8[outBase + 5] = sign_extend_nibble(b6 >> 4);
              tmpB_blk1_u8[outBase + 6] = sign_extend_nibble(b7 & 0x0F);
              tmpB_blk1_u8[outBase + 7] = sign_extend_nibble(b7 >> 4);
            }
          B_blk0_u8 = aie::load_v<MMUL::size_B>(tmpB_blk0_u8);
          B_blk1_u8 = aie::load_v<MMUL::size_B>(tmpB_blk1_u8);
          B01 = B_blk0_u8.pack();
          B11 = B_blk1_u8.pack();

          C00.mac(A01, B01);
          C01.mac(A01, B11);
          C10.mac(A11, B01);
          C11.mac(A11, B11);
          C20.mac(A21, B01);
          C21.mac(A21, B11);
          C30.mac(A31, B01);
          C31.mac(A31, B11);
        }

        aie::store_v(pC1, C00.template to_vector<int8>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<int8>());
        pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<int8>());
        pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<int8>());
        pC2 += MMUL::size_C;
        aie::store_v(pC3, C20.template to_vector<int8>());
        pC3 += MMUL::size_C;
        aie::store_v(pC3, C21.template to_vector<int8>());
        pC3 += MMUL::size_C;
        aie::store_v(pC4, C30.template to_vector<int8>());
        pC4 += MMUL::size_C;
        aie::store_v(pC4, C31.template to_vector<int8>());
        pC4 += MMUL::size_C;
      }
    }

  event1();
}

// int4 (input) x int8 (output) kernel using native 8b x 4b MMUL with shape
// 4x16x8
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x16x8_i4_i8(const int8 *__restrict pA,
                                                  const int8 *__restrict pB,
                                                  int8 *__restrict pC) {

  constexpr int r = 4;
  constexpr int s = 16;
  constexpr int t = 8;

  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  using MMUL = aie::mmul<r, s, t, int8, int4, accauto>;

  event0();

  for (unsigned z = 0; z < (m / r); z += 4)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      int8 *__restrict pC1 = pC + (z * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC2 = pC + ((z + 1) * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC3 = pC + ((z + 2) * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC4 = pC + ((z + 3) * (n / t) + 0) * MMUL::size_C;

      for (unsigned j = 0; j < (n / t); j += 2) {
        const int8 *__restrict pA1 = pA + (z * (k / s) + 0) * MMUL::size_A;
        const int8 *__restrict pA2 =
            pA + ((z + 1) * (k / s) + 0) * MMUL::size_A;
        const int8 *__restrict pA3 =
            pA + ((z + 2) * (k / s) + 0) * MMUL::size_A;
        const int8 *__restrict pA4 =
            pA + ((z + 3) * (k / s) + 0) * MMUL::size_A;

        const int8 *__restrict pB1 = pB + (0 * (n / t) + j) * MMUL::size_B;
        const int8 *__restrict pB2 =
            pB + (0 * (n / t) + (j + 1)) * MMUL::size_B;

        aie::vector<int8, MMUL::size_A> A01 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A11 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A21 = aie::load_v<MMUL::size_A>(pA3);
        pA3 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A31 = aie::load_v<MMUL::size_A>(pA4);
        pA4 += MMUL::size_A;

        aie::vector<int8, MMUL::size_B> B01_u8 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += (MMUL::size_B * (n / t));
        aie::vector<int8, MMUL::size_B> B11_u8 = aie::load_v<MMUL::size_B>(pB2);
        pB2 += (MMUL::size_B * (n / t));
        aie::vector<int4, MMUL::size_B> B01 = B01_u8.pack();
        aie::vector<int4, MMUL::size_B> B11 = B11_u8.pack();

        aie::vector<int8, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pC1);
        aie::vector<int8, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
        aie::vector<int8, MMUL::size_C> acc_C10 =
            aie::load_v<MMUL::size_C>(pC2);
        aie::vector<int8, MMUL::size_C> acc_C11 =
            aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
        aie::vector<int8, MMUL::size_C> acc_C20 =
            aie::load_v<MMUL::size_C>(pC3);
        aie::vector<int8, MMUL::size_C> acc_C21 =
            aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
        aie::vector<int8, MMUL::size_C> acc_C30 =
            aie::load_v<MMUL::size_C>(pC4);
        aie::vector<int8, MMUL::size_C> acc_C31 =
            aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);
        MMUL C20(acc_C20);
        MMUL C21(acc_C21);
        MMUL C30(acc_C30);
        MMUL C31(acc_C31);

        C00.mac(A01, B01);
        C01.mac(A01, B11);
        C10.mac(A11, B01);
        C11.mac(A11, B11);
        C20.mac(A21, B01);
        C21.mac(A21, B11);
        C30.mac(A31, B01);
        C31.mac(A31, B11);

        for (unsigned i = 1; i < (k / s); i += 1) {
          A01 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          A11 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          A21 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += MMUL::size_A;
          A31 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += MMUL::size_A;
          B01_u8 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += (MMUL::size_B * (n / t));
          B11_u8 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += (MMUL::size_B * (n / t));
          B01 = B01_u8.pack();
          B11 = B11_u8.pack();

          C00.mac(A01, B01);
          C01.mac(A01, B11);
          C10.mac(A11, B01);
          C11.mac(A11, B11);
          C20.mac(A21, B01);
          C21.mac(A21, B11);
          C30.mac(A31, B01);
          C31.mac(A31, B11);
        }

        aie::store_v(pC1, C00.template to_vector<int8>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<int8>());
        pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<int8>());
        pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<int8>());
        pC2 += MMUL::size_C;
        aie::store_v(pC3, C20.template to_vector<int8>());
        pC3 += MMUL::size_C;
        aie::store_v(pC3, C21.template to_vector<int8>());
        pC3 += MMUL::size_C;
        aie::store_v(pC4, C30.template to_vector<int8>());
        pC4 += MMUL::size_C;
        aie::store_v(pC4, C31.template to_vector<int8>());
        pC4 += MMUL::size_C;
      }
    }

  event1();
}

// (removed legacy 4x8x8 i4 path; use 4x16x8 variant instead)

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

// Explicit C binding for packed i4 case (row-major B)
void matmul_scalar_i8xi4_i8(int8 *a_in, int8 *b_in, int8 *c_out) {
  matmul_vectorized_4x32x8_i4_i8_packed<DIM_M, DIM_K, DIM_N>(a_in, b_in, c_out);
}

// Explicit C binding for packed i4 case (row-major B)
void matmul_i8xi4_i8(int8 *a_in, int8 *b_in, int8 *c_out) {
  matmul_vectorized_4x32x8_i4_i8_packed<DIM_M, DIM_K, DIM_N>(a_in, b_in, c_out);
}

} // extern "C"