# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.ir.types import uint64, uint256
from allo.utils import get_np_struct_type
import allo.dataflow as df
from allo.backend import hls

VLEN = 256
ENTRY_SIZE = 64
ELEN = 32

np_256 = get_np_struct_type(VLEN)


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


def test_vadd():

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def VEC(
            A: uint64[VLEN // ENTRY_SIZE],
            B: uint64[VLEN // ENTRY_SIZE],
            C: uint64[VLEN // ENTRY_SIZE],
        ):
            for i, j in allo.grid(
                VLEN // ENTRY_SIZE, ENTRY_SIZE // ELEN, name="vec_nest"
            ):
                C[i][j * ELEN : (j + 1) * ELEN] = (
                    A[i][j * ELEN : (j + 1) * ELEN] + B[i][j * ELEN : (j + 1) * ELEN]
                )

    A = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    B = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    packed_A = pack_i32_to_i256(A)
    packed_B = pack_i32_to_i256(B)
    packed_C = np.zeros((VLEN // ENTRY_SIZE,)).astype(np.uint64)

    mod = df.build(top, target="simulator")
    mod(packed_A, packed_B, packed_C)
    C_ = unpack_i256_to_i32(packed_C)
    np.testing.assert_allclose(A + B, C_, rtol=1e-5, atol=1e-5)
    print("PASSED!")

    s = df.customize(top)
    # unroll the lanes
    nest_loop_i = s.get_loops("VEC_0")["vec_nest"]["i"]
    s.unroll(nest_loop_i)
    nest_loop_j = s.get_loops("VEC_0")["vec_nest"]["j"]
    s.unroll(nest_loop_j)
    print(s.module)

    if hls.is_available("vitis_hls"):
        print("Starting hw Test...")
        modhw = s.build(
            target="vitis_hls",
            mode="hw",
            project=f"vec_hw.prj",
            wrap_io=False,
        )
        modhw(packed_A, packed_B, packed_C)
        C_ = unpack_i256_to_i32(packed_C)
        np.testing.assert_allclose(A + B, C_, rtol=1e-5, atol=1e-5)
        print("Passed hw Test!")


def test_vadd_adv():

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def VEC(
            A: uint256[1],
            B: uint256[1],
            C: uint256[1],
        ):
            for i in allo.grid(VLEN // ELEN, name="vec_nest"):
                C[0][i * ELEN : (i + 1) * ELEN] = (
                    A[0][i * ELEN : (i + 1) * ELEN] + B[0][i * ELEN : (i + 1) * ELEN]
                )

    A = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    B = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    C = np.zeros(
        VLEN // ELEN,
    ).astype(np.uint32)
    packed_A = np.ascontiguousarray(A).view(np_256)
    packed_B = np.ascontiguousarray(B).view(np_256)
    packed_C = np.ascontiguousarray(C).view(np_256)

    mod = df.build(top, target="simulator")
    mod(packed_A, packed_B, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(A + B, unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED!")

    s = df.customize(top)
    # unroll the lanes
    nest_loop_i = s.get_loops("VEC_0")["vec_nest"]["i"]
    s.unroll(nest_loop_i)
    print(s.module)

    if hls.is_available("vitis_hls"):
        print("Starting hw Test...")
        modhw = s.build(
            target="vitis_hls",
            mode="hw",
            project=f"vec_hw.prj",
            wrap_io=False,
        )
        modhw(packed_A, packed_B, packed_C)
        unpacked_C = packed_C.view(np.uint32)
        np.testing.assert_allclose(A + B, unpacked_C, rtol=1e-5, atol=1e-5)
        print("Passed hw Test!")


if __name__ == "__main__":
    # test_vadd()
    test_vadd_adv()
