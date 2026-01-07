//===- threshold.cc ----------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

enum _threshold_type {
  XF_THRESHOLD_TYPE_BINARY = 0,
  XF_THRESHOLD_TYPE_BINARY_INV = 1,
  XF_THRESHOLD_TYPE_TRUNC = 2,
  XF_THRESHOLD_TYPE_TOZERO = 3,
  XF_THRESHOLD_TYPE_TOZERO_INV = 4,
};

// #define THRESH_TYPE XF_THRESHOLD_TYPE_BINARY

#include <aie_api/aie.hpp>

template <typename T, int N>
__attribute__((noinline)) void
threshold_aie(T *img_in, T *img_out, const int32_t img_width,
              const int32_t img_height, const T &thresh_val, const T &max_val,
              const uint8_t thresholdType) {
  ::aie::vector<T, N> constants;
  ::aie::vector<T, N> data_out;
  ::aie::mask<N> temp_val;
  constants[0] = 0;          // updating constant zero_val value
  constants[1] = thresh_val; // updating constant threshold value
  constants[2] = max_val;    // updating constant max_val value

  switch (thresholdType) {
  case XF_THRESHOLD_TYPE_TRUNC:
    for (int j = 0; j < (img_height * img_width);
         j += N) // 16x samples per loop
      chess_prepare_for_pipelining chess_loop_range(14, ) {
        ::aie::vector<T, N> data_buf1 =
            ::aie::load_v(img_in); // in:00++15|_________|_________|_________
        img_in += N;
        data_out = ::aie::min(constants[1], data_buf1);
        ::aie::store_v(img_out, data_out);
        img_out += N;
      }
    break;
  case XF_THRESHOLD_TYPE_BINARY:
    for (int j = 0; j < (img_height * img_width);
         j += N) // 16x samples per loop
      chess_prepare_for_pipelining chess_loop_range(14, ) {
        ::aie::vector<T, N> data_buf1 =
            ::aie::load_v(img_in); // in:00++15|_________|_________|_________
        img_in += N;
        temp_val = ::aie::lt(constants[1], data_buf1);
        data_out = ::aie::select(constants[0], constants[2], temp_val);
        ::aie::store_v(img_out, data_out);
        img_out += N;
      }
    break;
  case XF_THRESHOLD_TYPE_BINARY_INV:
    for (int j = 0; j < (img_height * img_width);
         j += N) // 16x samples per loop
      chess_prepare_for_pipelining chess_loop_range(14, ) {
        ::aie::vector<T, N> data_buf1 =
            ::aie::load_v(img_in); // in:00++15|_________|_________|_________
        img_in += N;
        temp_val = ::aie::lt(constants[1], data_buf1);
        data_out = ::aie::select(constants[2], constants[0], temp_val);
        ::aie::store_v(img_out, data_out);
        img_out += N;
      }
    break;
  case XF_THRESHOLD_TYPE_TOZERO:
    for (int j = 0; j < (img_height * img_width);
         j += N) // 16x samples per loop
      chess_prepare_for_pipelining chess_loop_range(14, ) {
        ::aie::vector<T, N> data_buf1 =
            ::aie::load_v(img_in); // in:00++15|_________|_________|_________
        img_in += N;
        temp_val = ::aie::lt(constants[1], data_buf1);
        data_out = ::aie::select(constants[0], data_buf1, temp_val);
        ::aie::store_v(img_out, data_out);
        img_out += N;
      }
    break;
  case XF_THRESHOLD_TYPE_TOZERO_INV:
    for (int j = 0; j < (img_height * img_width);
         j += N) // 16x samples per loop
      chess_prepare_for_pipelining chess_loop_range(14, ) {
        ::aie::vector<T, N> data_buf1 =
            ::aie::load_v(img_in); // in:00++15|_________|_________|_________
        img_in += N;
        temp_val = ::aie::lt(constants[1], data_buf1);
        data_out = ::aie::select(data_buf1, constants[0], temp_val);
        ::aie::store_v(img_out, data_out);
        img_out += N;
      }
    break;
  default:
    for (int j = 0; j < (img_height * img_width);
         j += N) // 16x samples per loop
      chess_prepare_for_pipelining chess_loop_range(14, ) {
        ::aie::vector<T, N> data_buf1 =
            ::aie::load_v(img_in); // in:00++15|_________|_________|_________
        img_in += N;
        data_out = ::aie::min(constants[1], data_buf1);
        ::aie::store_v(img_out, data_out);
        img_out += N;
      }
  }
}

template <typename T, int N>
__attribute__((noinline)) void threshold4Ch_aie(
    T *img_in, T *img_out, const int32_t img_width, const int32_t img_height,
    const T &thresh_val1, const T &thresh_val2, const T &thresh_val3,
    const T &thresh_val4, const T &max_val1, const T &max_val2,
    const T &max_val3, const T &max_val4, const uint8_t thresholdType) {
  ::aie::vector<T, N> constants;
  ::aie::vector<T, N> data_out;
  ::aie::mask<N> temp_val;
  // constants[0] = 0;          // updating constant zero_val value
  // constants[1] = thresh_val; // updating constant threshold value
  // constants[2] = max_val;    // updating constant max_val value

  ::aie::vector<T, N> mask_zeros = ::aie::zeros<T, N>();
  ::aie::vector<T, N> mask_thresh;
  ::aie::vector<T, N> mask_max;
  for (int i = 0; i < N / 4; i++) {
    mask_thresh[i * 4] = thresh_val1;
    mask_thresh[i * 4 + 1] = thresh_val2;
    mask_thresh[i * 4 + 2] = thresh_val3;
    mask_thresh[i * 4 + 3] = thresh_val4;
    mask_max[i * 4] = max_val1;
    mask_max[i * 4 + 1] = max_val2;
    mask_max[i * 4 + 2] = max_val3;
    mask_max[i * 4 + 3] = max_val4;
  }

  switch (thresholdType) {
  case XF_THRESHOLD_TYPE_TRUNC:
    for (int j = 0; j < (img_height * img_width);
         j += N) // 16x samples per loop
      chess_prepare_for_pipelining chess_loop_range(14, ) {
        ::aie::vector<T, N> data_buf1 =
            ::aie::load_v(img_in); // in:00++15|_________|_________|_________
        img_in += N;
        data_out = ::aie::min(mask_thresh, data_buf1);
        ::aie::store_v(img_out, data_out);
        img_out += N;
      }
    break;
  case XF_THRESHOLD_TYPE_BINARY:
    for (int j = 0; j < (img_height * img_width);
         j += N) // 16x samples per loop
      chess_prepare_for_pipelining chess_loop_range(14, ) {
        ::aie::vector<T, N> data_buf1 =
            ::aie::load_v(img_in); // in:00++15|_________|_________|_________
        img_in += N;
        temp_val = ::aie::lt(mask_thresh, data_buf1);
        data_out = ::aie::select(mask_zeros, mask_max, temp_val);
        ::aie::store_v(img_out, data_out);
        img_out += N;
      }
    break;
  case XF_THRESHOLD_TYPE_BINARY_INV:
    for (int j = 0; j < (img_height * img_width);
         j += N) // 16x samples per loop
      chess_prepare_for_pipelining chess_loop_range(14, ) {
        ::aie::vector<T, N> data_buf1 =
            ::aie::load_v(img_in); // in:00++15|_________|_________|_________
        img_in += N;
        temp_val = ::aie::lt(mask_thresh, data_buf1);
        data_out = ::aie::select(mask_max, mask_zeros, temp_val);
        ::aie::store_v(img_out, data_out);
        img_out += N;
      }
    break;
  case XF_THRESHOLD_TYPE_TOZERO:
    for (int j = 0; j < (img_height * img_width);
         j += N) // 16x samples per loop
      chess_prepare_for_pipelining chess_loop_range(14, ) {
        ::aie::vector<T, N> data_buf1 =
            ::aie::load_v(img_in); // in:00++15|_________|_________|_________
        img_in += N;
        temp_val = ::aie::lt(mask_thresh, data_buf1);
        data_out = ::aie::select(mask_zeros, data_buf1, temp_val);
        ::aie::store_v(img_out, data_out);
        img_out += N;
      }
    break;
  case XF_THRESHOLD_TYPE_TOZERO_INV:
    for (int j = 0; j < (img_height * img_width);
         j += N) // 16x samples per loop
      chess_prepare_for_pipelining chess_loop_range(14, ) {
        ::aie::vector<T, N> data_buf1 =
            ::aie::load_v(img_in); // in:00++15|_________|_________|_________
        img_in += N;
        temp_val = ::aie::lt(mask_thresh, data_buf1);
        data_out = ::aie::select(data_buf1, mask_zeros, temp_val);
        ::aie::store_v(img_out, data_out);
        img_out += N;
      }
    break;
  default:
    for (int j = 0; j < (img_height * img_width);
         j += N) // 16x samples per loop
      chess_prepare_for_pipelining chess_loop_range(14, ) {
        ::aie::vector<T, N> data_buf1 =
            ::aie::load_v(img_in); // in:00++15|_________|_________|_________
        img_in += N;
        data_out = ::aie::min(mask_thresh, data_buf1);
        ::aie::store_v(img_out, data_out);
        img_out += N;
      }
  }
}

extern "C" {

void thresholdLine40(uint8_t in[1920], uint8_t out[1920]) {
  event0();
  threshold_aie<uint8_t, 64>(in, out, 1920, 1, 40, 255, 4);
  event1();
}

void thresholdLine30(uint8_t in[1920], uint8_t out[1920]) {
  event0();
  threshold_aie<uint8_t, 64>(in, out, 1920, 1, 30, 255, 0);
  event1();
}

void thresholdLine160(uint8_t in[1920], uint8_t out[1920]) {
  event0();
  threshold_aie<uint8_t, 64>(in, out, 1920, 1, 160, 255, 4);
  event1();
}

void thresholdLine90(uint8_t in[1920], uint8_t out[1920]) {
  event0();
  threshold_aie<uint8_t, 64>(in, out, 1920, 1, 90, 255, 0);
  event1();
}
} // extern "C"