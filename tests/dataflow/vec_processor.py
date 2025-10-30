# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.ir.types import UInt, uint2, int32
import allo.dataflow as df

VLEN = 128
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

def pack_i32_to_i128(arr):
    arr = np.array(arr, dtype=np.uint32)
    n = len(arr)
    assert n % 4 == 0

    packed = []
    for i in range(0, n, 4):
        value = 0
        for j in range(4):
            value |= int(arr[i + j]) << (32 * j)
        packed.append(value)
    return np.array(packed, dtype=object)


def unpack_i128_to_i32(value):
    result = []
    mask = (1 << 32) - 1
    for i in range(4):
        part = (value >> (32 * i)) & mask
        result.append(np.int32(part))
    return np.array(result, dtype=np.int32)


@df.region()
def top():
    @df.kernel(mapping=[1])
    def gemm0(A: UInt(VLEN)[1], B: UInt(VLEN)[1], C: UInt(VLEN)[1]):
        for i in range(VLEN // ELEN):
            C[0][i * ELEN : (i + 1) * ELEN] = (
                A[0][i * ELEN : (i + 1) * ELEN] + B[0][i * ELEN : (i + 1) * ELEN]
            )


A = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.int32)
B = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.int32)
packed_A = pack_i32_to_i128(A)
packed_B = pack_i32_to_i128(B)
packed_C = pack_i32_to_i128(B)

# mod = df.build(top)
mod = df.build(top, target="simulator")
mod(packed_A, packed_B, packed_C)
C_ = unpack_i128_to_i32(packed_C[0])
print(C_)
print(A + B, C_)
