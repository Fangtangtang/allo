# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import UInt
import allo.dataflow as df

VLEN = 256
ELEN = 32

import numpy as np


def pack_i32_to_i256(arr):
    arr = np.array(arr, dtype=np.uint32)
    n = len(arr)
    assert n % 8 == 0

    packed = []
    for i in range(0, n, 8):
        value = 0
        for j in range(8):
            value |= int(arr[i + j]) << (32 * j)
        packed.append(value)
    return np.array(packed, dtype=object)


def unpack_i256_to_i32(value):
    result = []
    mask = (1 << 32) - 1
    for i in range(8):
        part = (value >> (32 * i)) & mask
        result.append(np.int32(part))
    return np.array(result, dtype=np.int32)


@df.region()
def top():
    @df.kernel(mapping=[1])
    def gemm0(A: UInt(VLEN), B: UInt(VLEN), C: UInt(VLEN)):
        with allo.meta_for(VLEN // ELEN) as i:
            C[i * ELEN : (i + 1) * ELEN] = (
                A[i * ELEN : (i + 1) * ELEN] + B[i * ELEN : (i + 1) * ELEN]
            )


A = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.int32)
B = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.int32)
packed_A = pack_i32_to_i256(A)[0]
packed_B = pack_i32_to_i256(B)[0]
packed_C = pack_i32_to_i256(B)[0]

mod = df.build(top, target="simulator")
mod(packed_A, packed_B, packed_C)
C_ = unpack_i256_to_i32(packed_C)
print(A + B, C_)
