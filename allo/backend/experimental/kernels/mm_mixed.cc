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

#define PACKED_B_SKIP_UNPACK 0

// v2: packed B
// int4 (input) x int8 (output) kernel, B is packed (2×int4 per byte), shape
// 4x16x8
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x16x8_i4_i8_packedB_v2(
    const int8 *__restrict pA, const int8 *__restrict pB /* packed bytes */,
    int8 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 16;
  constexpr int t = 8;

  using MMUL = aie::mmul<r, s, t, int8, int4, accauto>;
  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);
  static_assert((n % 2) == 0, "n must be even because B is packed along N");

  // Unpack: expand the packed byte stream to int8 values in [-8, 7], then
  // pack() into an int4 vector
  auto unpack_i4_bytes_to_i8 = [](const int8 *srcPacked,
                                  int8 *dstExpanded /* len = MMUL::size_B */) {
    constexpr unsigned BYTES = MMUL::size_B / 2; // two int4 values per byte
    const uint8_t *p = reinterpret_cast<const uint8_t *>(srcPacked);
    for (unsigned i = 0; i < BYTES; ++i) {
      uint8_t b = p[i];
      // Map 4-bit unsigned 0..15 to signed -8..7: (x ^ 0x8) - 0x8
      int8 lo = static_cast<int8>(((b & 0x0F) ^ 0x08) - 0x08); // column 2p
      int8 hi =
          static_cast<int8>((((b >> 4) & 0x0F) ^ 0x08) - 0x08); // column 2p+1
      dstExpanded[2 * i + 0] = lo;
      dstExpanded[2 * i + 1] = hi;
    }
  };

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

        // Note: base addresses and strides are based on "byte count = element
        // count / 2"
        const int8 *__restrict pB1 =
            pB + (0 * (n / t) + j) * (MMUL::size_B / 2);
        const int8 *__restrict pB2 =
            pB + (0 * (n / t) + (j + 1)) * (MMUL::size_B / 2);

        aie::vector<int8, MMUL::size_A> A01 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A11 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A21 = aie::load_v<MMUL::size_A>(pA3);
        pA3 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A31 = aie::load_v<MMUL::size_A>(pA4);
        pA4 += MMUL::size_A;

        // Unpack two B tiles (corresponding to j and j+1)
        alignas(32) int8 B01_exp[MMUL::size_B];
        alignas(32) int8 B11_exp[MMUL::size_B];
#if !PACKED_B_SKIP_UNPACK
        unpack_i4_bytes_to_i8(pB1, B01_exp);
        unpack_i4_bytes_to_i8(pB2, B11_exp);
#else
        for (unsigned u = 0; u < MMUL::size_B; ++u) {
          B01_exp[u] = 0;
          B11_exp[u] = 0;
        }
#endif
        // Offset to the next block along K: previously MMUL::size_B*(n/t), now
        // halved
        pB1 += (MMUL::size_B / 2) * (n / t);
        pB2 += (MMUL::size_B / 2) * (n / t);

        aie::vector<int8, MMUL::size_B> B01_u8 =
            aie::load_v<MMUL::size_B>(B01_exp);
        aie::vector<int8, MMUL::size_B> B11_u8 =
            aie::load_v<MMUL::size_B>(B11_exp);
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

        MMUL C00(acc_C00), C01(acc_C01), C10(acc_C10), C11(acc_C11);
        MMUL C20(acc_C20), C21(acc_C21), C30(acc_C30), C31(acc_C31);

        C00.mac(A01, B01);
        C01.mac(A01, B11);
        C10.mac(A11, B01);
        C11.mac(A11, B11);
        C20.mac(A21, B01);
        C21.mac(A21, B11);
        C30.mac(A31, B01);
        C31.mac(A31, B11);

        for (unsigned i = 1; i < (k / s); ++i) {
          A01 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          A11 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          A21 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += MMUL::size_A;
          A31 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += MMUL::size_A;

#if !PACKED_B_SKIP_UNPACK
          unpack_i4_bytes_to_i8(pB1, B01_exp);
          unpack_i4_bytes_to_i8(pB2, B11_exp);
#else
          for (unsigned u = 0; u < MMUL::size_B; ++u) {
            B01_exp[u] = 0;
            B11_exp[u] = 0;
          }
#endif
          pB1 += (MMUL::size_B / 2) * (n / t);
          pB2 += (MMUL::size_B / 2) * (n / t);

          B01_u8 = aie::load_v<MMUL::size_B>(B01_exp);
          B11_u8 = aie::load_v<MMUL::size_B>(B11_exp);
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

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x16x8_i4_i8_packedB_v4(
    const int8 *__restrict pA, const int8 *__restrict pB, int8 *__restrict pC) {

  constexpr int r = 4, s = 16, t = 8;
  using MMUL = aie::mmul<r, s, t, int8, int4, accauto>;
  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  event0();

  for (unsigned z = 0; z < (m / r); z += 4)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      int8 *__restrict pC1 = pC + (z * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC2 = pC + ((z + 1) * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC3 = pC + ((z + 2) * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC4 = pC + ((z + 3) * (n / t) + 0) * MMUL::size_C;

      for (unsigned j = 0; j < (n / t); j += 2) {
        // A pointers are the same as in the original version
        const int8 *__restrict pA1 = pA + (z * (k / s) + 0) * MMUL::size_A;
        const int8 *__restrict pA2 =
            pA + ((z + 1) * (k / s) + 0) * MMUL::size_A;
        const int8 *__restrict pA3 =
            pA + ((z + 2) * (k / s) + 0) * MMUL::size_A;
        const int8 *__restrict pA4 =
            pA + ((z + 3) * (k / s) + 0) * MMUL::size_A;

        // Key: B is now tile-major. Each (s×t) tile is a contiguous 64B block
        // in L1. pB1 points to tile j, pB2 points to tile (j+1); advance by 64B
        // per K-chunk.
        const uint8_t *__restrict pB1 = reinterpret_cast<const uint8_t *>(pB) +
                                        (0 * (n / t) + j) * (MMUL::size_B / 2);
        const uint8_t *__restrict pB2 =
            reinterpret_cast<const uint8_t *>(pB) +
            (0 * (n / t) + (j + 1)) * (MMUL::size_B / 2);

        aie::vector<int8, MMUL::size_A> A01 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A11 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A21 = aie::load_v<MMUL::size_A>(pA3);
        pA3 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A31 = aie::load_v<MMUL::size_A>(pA4);
        pA4 += MMUL::size_A;

        // === Load a 64B packed tile at once ===
        aie::vector<uint8_t, s * t / 2> Bp0 = aie::load_v<s * t / 2>(pB1);
        aie::vector<uint8_t, s * t / 2> Bp1 = aie::load_v<s * t / 2>(pB2);

        // directly vector_cast it to int4
        auto B01 = aie::vector_cast<int4>(Bp0);
        auto B11 = aie::vector_cast<int4>(Bp1);

        // C path same as the original version
        auto acc_C00 = aie::load_v<MMUL::size_C>(pC1);
        auto acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
        auto acc_C10 = aie::load_v<MMUL::size_C>(pC2);
        auto acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
        auto acc_C20 = aie::load_v<MMUL::size_C>(pC3);
        auto acc_C21 = aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
        auto acc_C30 = aie::load_v<MMUL::size_C>(pC4);
        auto acc_C31 = aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);

        MMUL C00(acc_C00), C01(acc_C01), C10(acc_C10), C11(acc_C11);
        MMUL C20(acc_C20), C21(acc_C21), C30(acc_C30), C31(acc_C31);

        C00.mac(A01, B01);
        C01.mac(A01, B11);
        C10.mac(A11, B01);
        C11.mac(A11, B11);
        C20.mac(A21, B01);
        C21.mac(A21, B11);
        C30.mac(A31, B01);
        C31.mac(A31, B11);

        // Remaining blocks along K
        for (unsigned i = 1; i < (k / s); ++i) {
          A01 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          A11 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          A21 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += MMUL::size_A;
          A31 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += MMUL::size_A;

          pB1 += (MMUL::size_B / 2) * (n / t);
          ; // next tile along K, step 64B
          pB2 += (MMUL::size_B / 2) * (n / t);
          ;

          Bp0 = aie::load_v<s * t / 2>(pB1);
          Bp1 = aie::load_v<s * t / 2>(pB2);

          B01 = aie::vector_cast<int4>(Bp0);
          B11 = aie::vector_cast<int4>(Bp1);

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

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x16x8_i4_packedA_i4_packedB(
    const int8 *__restrict pA, const int8 *__restrict pB, int8 *__restrict pC) {

  constexpr int r = 4, s = 16, t = 8;
  using MMUL = aie::mmul<r, s, t, int8, int4, accauto>;
  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  event0();

  for (unsigned z = 0; z < (m / r); z += 4)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      int8 *__restrict pC1 = pC + (z * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC2 = pC + ((z + 1) * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC3 = pC + ((z + 2) * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC4 = pC + ((z + 3) * (n / t) + 0) * MMUL::size_C;

      for (unsigned j = 0; j < (n / t); j += 2) {
        // A pointers are the same as in the original version
        const int8 *__restrict pA1 =
            pA + (z * (k / s) + 0) * (MMUL::size_A / 2);
        const int8 *__restrict pA2 =
            pA + ((z + 1) * (k / s) + 0) * (MMUL::size_A / 2);
        const int8 *__restrict pA3 =
            pA + ((z + 2) * (k / s) + 0) * (MMUL::size_A / 2);
        const int8 *__restrict pA4 =
            pA + ((z + 3) * (k / s) + 0) * (MMUL::size_A / 2);

        // Key: B is tile-major. Each (s×t) tile is a contiguous 64B block in
        // L1. pB1 points to tile j, pB2 points to tile (j+1); advance 64B per
        // K-chunk.
        const uint8_t *__restrict pB1 = reinterpret_cast<const uint8_t *>(pB) +
                                        (0 * (n / t) + j) * (MMUL::size_B / 2);
        const uint8_t *__restrict pB2 =
            reinterpret_cast<const uint8_t *>(pB) +
            (0 * (n / t) + (j + 1)) * (MMUL::size_B / 2);

        // use vector_cast to load (MMUL::size_A / 2) int8 data as
        // (MMUL::size_A) int4; then unpack() to int8
        auto A01_i4 = aie::vector_cast<int4>(aie::load_v<r * s / 2>(pA1));
        aie::vector<int8, MMUL::size_A> A01 = aie::unpack(A01_i4);
        auto A11_i4 = aie::vector_cast<int4>(aie::load_v<r * s / 2>(pA2));
        aie::vector<int8, MMUL::size_A> A11 = aie::unpack(A11_i4);
        auto A21_i4 = aie::vector_cast<int4>(aie::load_v<r * s / 2>(pA3));
        aie::vector<int8, MMUL::size_A> A21 = aie::unpack(A21_i4);
        auto A31_i4 = aie::vector_cast<int4>(aie::load_v<r * s / 2>(pA4));
        aie::vector<int8, MMUL::size_A> A31 = aie::unpack(A31_i4);

        // === Load a 64B packed tile at once ===
        aie::vector<uint8_t, s * t / 2> Bp0 = aie::load_v<s * t / 2>(pB1);
        aie::vector<uint8_t, s * t / 2> Bp1 = aie::load_v<s * t / 2>(pB2);

        // directly vector_cast it to int4
        auto B01 = aie::vector_cast<int4>(Bp0);
        auto B11 = aie::vector_cast<int4>(Bp1);

        // C path same as the original version
        auto acc_C00 = aie::load_v<MMUL::size_C>(pC1);
        auto acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
        auto acc_C10 = aie::load_v<MMUL::size_C>(pC2);
        auto acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
        auto acc_C20 = aie::load_v<MMUL::size_C>(pC3);
        auto acc_C21 = aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
        auto acc_C30 = aie::load_v<MMUL::size_C>(pC4);
        auto acc_C31 = aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);

        MMUL C00(acc_C00), C01(acc_C01), C10(acc_C10), C11(acc_C11);
        MMUL C20(acc_C20), C21(acc_C21), C30(acc_C30), C31(acc_C31);

        C00.mac(A01, B01);
        C01.mac(A01, B11);
        C10.mac(A11, B01);
        C11.mac(A11, B11);
        C20.mac(A21, B01);
        C21.mac(A21, B11);
        C30.mac(A31, B01);
        C31.mac(A31, B11);

        // Remaining blocks along K
        for (unsigned i = 1; i < (k / s); ++i) {
          pA1 += MMUL::size_A / 2;
          pA2 += MMUL::size_A / 2;
          pA3 += MMUL::size_A / 2;
          pA4 += MMUL::size_A / 2;

          A01_i4 = aie::vector_cast<int4>(aie::load_v<MMUL::size_A / 2>(pA1));
          A01 = aie::unpack(A01_i4);
          A11_i4 = aie::vector_cast<int4>(aie::load_v<MMUL::size_A / 2>(pA2));
          A11 = aie::unpack(A11_i4);
          A21_i4 = aie::vector_cast<int4>(aie::load_v<MMUL::size_A / 2>(pA3));
          A21 = aie::unpack(A21_i4);
          A31_i4 = aie::vector_cast<int4>(aie::load_v<MMUL::size_A / 2>(pA4));
          A31 = aie::unpack(A31_i4);

          pB1 += (MMUL::size_B / 2) * (n / t);
          ; // next tile along K, step 64B
          pB2 += (MMUL::size_B / 2) * (n / t);
          ;

          Bp0 = aie::load_v<s * t / 2>(pB1);
          Bp1 = aie::load_v<s * t / 2>(pB2);

          B01 = aie::vector_cast<int4>(Bp0);
          B11 = aie::vector_cast<int4>(Bp1);

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

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie:mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this particular kernel we expand the aie::mmul two times in each
 * input matrices A (in 'm' dimension, or rowA) and B (in 'n' dimension, or
 * ColB), leading to a 2x2 expansion in output matrix C (see C00, C01, C10, C11
 * below). This expansion helps with accumulator registers usage, which leads in
 * attaining high kernel efficiency (SIMD utilization).
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order, as
 * shown in the example below for matrix A:
 *
 *      <-s->
 *    _  ________________________
 * 	  r |  1 |  2 |  3 | ...
 * 	  _ |____|____|____|
 * 	    |  x | x+1| x+2| ...
 * 	    |____|____|____|
 * 	    |.
 * 	    |.
 * 	    |.
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_2x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      T_out *__restrict pC1 = pC + (z * colB + 0) * MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1) * colB + 0) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1) * colA + 0) * MMUL::size_A;
          const T_in *__restrict pB1 = pB + (0 * colB + j) * MMUL::size_B;
          const T_in *__restrict pB2 = pB + (0 * colB + (j + 1)) * MMUL::size_B;

          aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * colB;
          aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += MMUL::size_B * colB;

          // Load partial results from C buffer for accumulation in-place. The
          // zero.cc function handles the zeroing of data when a new
          // accumulation is needed (after the 'K' reduction dimension)
          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C10(acc_C10);
          MMUL C11(acc_C11);

          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);

          for (unsigned i = 1; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              A0 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += MMUL::size_A;
              A1 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += MMUL::size_A;
              B0 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B * colB;
              B1 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B * colB;

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);
            }

          // TODO make shift right here to keep most significat bits
          // when lowering the output
          // example below shows how to shift right 10 bits
          // #define SHIFT 10
          // aie::store_v(pC1, C00.template to_vector<T_out>(SHIFT));
          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C;
        }
    }

  event1();
}

/* Similar to the kernel above, but we expand matrix A (in 'm' dimension, or
 * rowA) 4 times, while matrix B is expanded 2 times (in 'n' dimension, or
 * ColB). This is very helpful in attaining high kernel efficiency for some
 * precisions (e.g., int8)
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_4x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 4)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      T_out *__restrict pC1 = pC + (z * colB + 0) * MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1) * colB + 0) * MMUL::size_C;
      T_out *__restrict pC3 = pC + ((z + 2) * colB + 0) * MMUL::size_C;
      T_out *__restrict pC4 = pC + ((z + 3) * colB + 0) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1) * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA3 = pA + ((z + 2) * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA4 = pA + ((z + 3) * colA + 0) * MMUL::size_A;

          const T_in *__restrict pB1 = pB + (0 * colB + j) * MMUL::size_B;
          const T_in *__restrict pB2 = pB + (0 * colB + (j + 1)) * MMUL::size_B;

          aie::vector<T_in, MMUL::size_A> A01 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A11 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A21 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A31 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += (MMUL::size_B * colB);
          aie::vector<T_in, MMUL::size_B> B11 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += (MMUL::size_B * colB);

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C20 =
              aie::load_v<MMUL::size_C>(pC3);
          aie::vector<T_out, MMUL::size_C> acc_C21 =
              aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C30 =
              aie::load_v<MMUL::size_C>(pC4);
          aie::vector<T_out, MMUL::size_C> acc_C31 =
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

          for (unsigned i = 1; i < colA; i += 1)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              A01 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += MMUL::size_A;
              A11 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += MMUL::size_A;
              A21 = aie::load_v<MMUL::size_A>(pA3);
              pA3 += MMUL::size_A;
              A31 = aie::load_v<MMUL::size_A>(pA4);
              pA4 += MMUL::size_A;
              B01 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += (MMUL::size_B * colB);
              B11 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += (MMUL::size_B * colB);

              C00.mac(A01, B01);
              C01.mac(A01, B11);
              C10.mac(A11, B01);
              C11.mac(A11, B11);
              C20.mac(A21, B01);
              C21.mac(A21, B11);
              C30.mac(A31, B01);
              C31.mac(A31, B11);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC3, C20.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC3, C21.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC4, C30.template to_vector<T_out>());
          pC4 += MMUL::size_C;
          aie::store_v(pC4, C31.template to_vector<T_out>());
          pC4 += MMUL::size_C;
        }
    }

  event1();
}

/* Similar to the kernel aboves, we expand matrix A (in 'm' dimension, or rowA)
 * 4 times, while matrix B is expanded spatially 4 times (in 'n' dimension, or
 * ColB), for even higher accumulator usage. This is very helpful in attaining
 * high kernel efficiency for some precisions (e.g., bf16)
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_4x4(const T_in *__restrict pA,
                                         const T_in *__restrict pB,
                                         T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 4)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + (z * colB + 0) * MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1) * colB + 0) * MMUL::size_C;
      T_out *__restrict pC3 = pC + ((z + 2) * colB + 0) * MMUL::size_C;
      T_out *__restrict pC4 = pC + ((z + 3) * colB + 0) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 4)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1) * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA3 = pA + ((z + 2) * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA4 = pA + ((z + 3) * colA + 0) * MMUL::size_A;

          const T_in *__restrict pB1 = pB + (0 * colB + j) * MMUL::size_B;
          const T_in *__restrict pB2 = pB + (0 * colB + (j + 1)) * MMUL::size_B;
          const T_in *__restrict pB3 = pB + (0 * colB + (j + 2)) * MMUL::size_B;
          const T_in *__restrict pB4 = pB + (0 * colB + (j + 3)) * MMUL::size_B;

          aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A2 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A3 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * colB;
          aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += MMUL::size_B * colB;
          aie::vector<T_in, MMUL::size_B> B2 = aie::load_v<MMUL::size_B>(pB3);
          pB3 += MMUL::size_B * colB;
          aie::vector<T_in, MMUL::size_B> B3 = aie::load_v<MMUL::size_B>(pB4);
          pB4 += MMUL::size_B * colB;

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C02 =
              aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C03 =
              aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C);

          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C12 =
              aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C13 =
              aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C);

          aie::vector<T_out, MMUL::size_C> acc_C20 =
              aie::load_v<MMUL::size_C>(pC3);
          aie::vector<T_out, MMUL::size_C> acc_C21 =
              aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C22 =
              aie::load_v<MMUL::size_C>(pC3 + 2 * MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C23 =
              aie::load_v<MMUL::size_C>(pC3 + 3 * MMUL::size_C);

          aie::vector<T_out, MMUL::size_C> acc_C30 =
              aie::load_v<MMUL::size_C>(pC4);
          aie::vector<T_out, MMUL::size_C> acc_C31 =
              aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C32 =
              aie::load_v<MMUL::size_C>(pC4 + 2 * MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C33 =
              aie::load_v<MMUL::size_C>(pC4 + 3 * MMUL::size_C);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C02(acc_C02);
          MMUL C03(acc_C03);

          MMUL C10(acc_C10);
          MMUL C11(acc_C11);
          MMUL C12(acc_C12);
          MMUL C13(acc_C13);

          MMUL C20(acc_C20);
          MMUL C21(acc_C21);
          MMUL C22(acc_C22);
          MMUL C23(acc_C23);

          MMUL C30(acc_C30);
          MMUL C31(acc_C31);
          MMUL C32(acc_C32);
          MMUL C33(acc_C33);

          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);

          C02.mac(A0, B2);
          C03.mac(A0, B3);
          C12.mac(A1, B2);
          C13.mac(A1, B3);

          C20.mac(A2, B0);
          C21.mac(A2, B1);
          C30.mac(A3, B0);
          C31.mac(A3, B1);

          C22.mac(A2, B2);
          C23.mac(A2, B3);
          C32.mac(A3, B2);
          C33.mac(A3, B3);

          for (unsigned i = 1; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              A0 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += MMUL::size_A;
              A1 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += MMUL::size_A;
              A2 = aie::load_v<MMUL::size_A>(pA3);
              pA3 += MMUL::size_A;
              A3 = aie::load_v<MMUL::size_A>(pA4);
              pA4 += MMUL::size_A;

              B0 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B * colB;
              B1 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B * colB;
              B2 = aie::load_v<MMUL::size_B>(pB3);
              pB3 += MMUL::size_B * colB;
              B3 = aie::load_v<MMUL::size_B>(pB4);
              pB4 += MMUL::size_B * colB;

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);

              C02.mac(A0, B2);
              C03.mac(A0, B3);
              C12.mac(A1, B2);
              C13.mac(A1, B3);

              C20.mac(A2, B0);
              C21.mac(A2, B1);
              C30.mac(A3, B0);
              C31.mac(A3, B1);

              C22.mac(A2, B2);
              C23.mac(A2, B3);
              C32.mac(A3, B2);
              C33.mac(A3, B3);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C02.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C03.template to_vector<T_out>());
          pC1 += MMUL::size_C;

          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C12.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C13.template to_vector<T_out>());
          pC2 += MMUL::size_C;

          aie::store_v(pC3, C20.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC3, C21.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC3, C22.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC3, C23.template to_vector<T_out>());
          pC3 += MMUL::size_C;

          aie::store_v(pC4, C30.template to_vector<T_out>());
          pC4 += MMUL::size_C;
          aie::store_v(pC4, C31.template to_vector<T_out>());
          pC4 += MMUL::size_C;
          aie::store_v(pC4, C32.template to_vector<T_out>());
          pC4 += MMUL::size_C;
          aie::store_v(pC4, C33.template to_vector<T_out>());
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
void matmul_i8xi4_i8(int8 *a_in, int8 *b_in, int8 *c_out) {
  // v1: pretty slow
  // matmul_vectorized_4x32x8_i4_i8_packed<DIM_M, DIM_K, DIM_N>(a_in, b_in,
  // c_out); matmul_vectorized_4x16x8_i4_i8_packedB_v2<DIM_M, DIM_K,
  // DIM_N>(a_in, b_in, c_out);
  matmul_vectorized_4x16x8_i4_i8_packedB_v4<DIM_M, DIM_K, DIM_N>(a_in, b_in,
                                                                 c_out);
}

// int4xint4, both A and B are packed
void matmul_i4xi4_i8(int8 *a_in, int8 *b_in, int8 *c_out) {
  matmul_vectorized_4x16x8_i4_packedA_i4_packedB<DIM_M, DIM_K, DIM_N>(
      a_in, b_in, c_out);
}

} // extern "C"