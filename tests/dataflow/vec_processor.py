# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.ir.types import uint64
import allo.dataflow as df

VLEN = 256
ENTRY_SIZE = 64
ELEN = 32

import sys

# def kernel(A: uint2[10], B: int32[10]):
#     for i in range(10):
#         B[i][0:2] = A[i]
#         B[i][2:4] = A[i]

# s = allo.customize(kernel)
# np_A = np.random.randint(0, 4, size=(10,))
# np_B = np.zeros_like(np_A)
# # np_A = np.random.randint(2, size=(10,))
# # np_B = np.random.randint(7, size=(10,))
# golden = (np_A) | (np_A << 2)
# mod = s.build()
# mod(np_A, np_B)
# assert np.array_equal(golden, np_B)
# sys.exit(0)


def pack_i32_to_i256(arr):
    arr = np.array(arr, dtype=np.uint32)
    n = len(arr)
    assert n % 8 == 0, "Length must be a multiple of 8"

    packed = []
    for i in range(0, n, 8):
        parts = []
        for j in range(0, 8, 2):
            low32 = int(arr[i + j])
            high32 = int(arr[i + j + 1])
            val64 = (high32 << 32) | low32
            parts.append(np.uint64(val64))
        packed.append(parts)
    return np.array(packed, dtype=np.uint64)


def unpack_i256_to_i32(packed):
    packed = np.array(packed, dtype=np.int64)
    result = []
    mask = (1 << 32) - 1
    for j in range(4):
        val64 = int(packed[j])
        low32 = np.uint32(val64 & mask)
        high32 = np.uint32((val64 >> 32) & mask)
        result.extend([low32, high32])
    return np.array(result, dtype=np.uint32)


@df.region()
def top():
    @df.kernel(mapping=[1])
    def gemm0(
        A: uint64[VLEN // ENTRY_SIZE],
        B: uint64[VLEN // ENTRY_SIZE],
        C: uint64[VLEN // ENTRY_SIZE],
    ):
        for i in range(VLEN // ENTRY_SIZE):
            for j in range(ENTRY_SIZE // ELEN):
                C[i][j * ELEN : (j + 1) * ELEN] = (
                    A[i][j * ELEN : (j + 1) * ELEN] + B[i][j * ELEN : (j + 1) * ELEN]
                )


A = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
B = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
packed_A = pack_i32_to_i256(A)
packed_B = pack_i32_to_i256(B)
packed_C = np.zeros((VLEN // ENTRY_SIZE,)).astype(np.uint64)

# mod = df.build(top)
mod = df.build(top, target="simulator")
mod(packed_A, packed_B, packed_C)
C_ = unpack_i256_to_i32(packed_C)
np.testing.assert_allclose(A + B, C_, rtol=1e-5, atol=1e-5)
print("PASSED!")
