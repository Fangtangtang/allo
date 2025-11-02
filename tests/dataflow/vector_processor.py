# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import uint256, uint32, uint8, bool, int8, int16, int32
import allo.dataflow as df
from allo.utils import get_np_struct_type
from allo.backend import hls

VLEN = 256
ELEN = 32

import numpy as np

np_256 = get_np_struct_type(VLEN)


def test_vadd():
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
    C = np.zeros(VLEN // ELEN).astype(np.uint32)
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
        print("Starting Test...")
        mod = s.build(
            target="vitis_hls",
            mode="hw_emu",
            project=f"vec_adv_hw_emu.prj",
            wrap_io=False,
        )
        mod(packed_A, packed_B, packed_C)
        unpacked_C = packed_C.view(np.uint32)
        np.testing.assert_allclose(A + B, unpacked_C, rtol=1e-5, atol=1e-5)
        print(unpacked_C)
        print("Passed Test!")


# ############################
# Scalar type
# ############################
EEW8 = 0  # 8bits
EEW16 = 1  # 16bits
EEW32 = 2  # 32bits


# ############################
# Vector Instructions
# ############################
# - min/max
# VMAX_VV = 0b0000
# VMAX_VX = 0b0001
# VMIN_VV = 0b0010
# VMIN_VX = 0b0011
VMAX = 0
VMIN = 1

# - add/sub
# VADD_VV = 0b0100
# VADD_VX = 0b0101
# # VADD_VI = 6
# VSUB_VV = 0b0110
# VSUB_VX = 0b0111
VADD = 2
VSUB = 3

# - mul
# VMUL_VV = 0b1000
# VMUL_VX = 0b1001
VMUL = 4
# ############################


def test_vec():
    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def VEC(
            inst: uint8[1],
            ele_type: uint8[1],
            vv: bool[1],
            vs1: uint256[1],
            vs2: uint256[1],
            rs1: uint32[1],
            vd: uint256[1],
        ):
            # prepare operand
            operand1: uint256
            operand2: uint256 = vs2[0]
            if vv[0]:
                operand1 = vs1[0]
            else:
                if ele_type[0] == EEW8:
                    for i in allo.grid(VLEN // 8, name="scalar_to_vector_8"):
                        operand1[i * 8 : (i + 1) * 8] = rs1[0][0:8]
                elif ele_type[0] == EEW16:
                    for i in allo.grid(VLEN // 16, name="scalar_to_vector_16"):
                        operand1[i * 16 : (i + 1) * 16] = rs1[0][0:16]
                elif ele_type[0] == EEW32:
                    for i in allo.grid(VLEN // 32, name="scalar_to_vector_32"):
                        operand1[i * 32 : (i + 1) * 32] = rs1[0]
            # compute
            if inst[0] == VMAX or inst[0] == VMIN:
                if ele_type[0] == EEW8:
                    for i in allo.grid(VLEN // 8, name="min_max_8"):
                        scalar1: int8 = operand1[i * 8 : (i + 1) * 8]
                        scalar2: int8 = operand2[i * 8 : (i + 1) * 8]
                        condition: bool = scalar1 > scalar2
                        if inst[0] == VMAX:
                            if condition:
                                vd[0][i * 8 : (i + 1) * 8] = scalar1
                            else:
                                vd[0][i * 8 : (i + 1) * 8] = scalar2
                        else:
                            if condition:
                                vd[0][i * 8 : (i + 1) * 8] = scalar2
                            else:
                                vd[0][i * 8 : (i + 1) * 8] = scalar1
                elif ele_type[0] == EEW16:
                    for i in allo.grid(VLEN // 16, name="min_max_16"):
                        scalar1: int16 = operand1[i * 16 : (i + 1) * 16]
                        scalar2: int16 = operand2[i * 16 : (i + 1) * 16]
                        condition: bool = scalar1 > scalar2
                        if inst[0] == VMAX:
                            if condition:
                                vd[0][i * 16 : (i + 1) * 16] = scalar1
                            else:
                                vd[0][i * 16 : (i + 1) * 16] = scalar2
                        else:
                            if condition:
                                vd[0][i * 16 : (i + 1) * 16] = scalar2
                            else:
                                vd[0][i * 16 : (i + 1) * 16] = scalar1
                elif ele_type[0] == EEW32:
                    for i in allo.grid(VLEN // 32, name="min_max_32"):
                        scalar1: int32 = operand1[i * 32 : (i + 1) * 32]
                        scalar2: int32 = operand2[i * 32 : (i + 1) * 32]
                        condition: bool = scalar1 > scalar2
                        if inst[0] == VMAX:
                            if condition:
                                vd[0][i * 32 : (i + 1) * 32] = scalar1
                            else:
                                vd[0][i * 32 : (i + 1) * 32] = scalar2
                        else:
                            if condition:
                                vd[0][i * 32 : (i + 1) * 32] = scalar2
                            else:
                                vd[0][i * 32 : (i + 1) * 32] = scalar1
            else:
                if ele_type[0] == EEW8:
                    for i in allo.grid(VLEN // 8, name="arith_8"):
                        scalar1: int8 = operand1[i * 8 : (i + 1) * 8]
                        scalar2: int8 = operand2[i * 8 : (i + 1) * 8]
                        if inst[0] == VADD:
                            vd[0][i * 8 : (i + 1) * 8] = scalar1 + scalar2
                        elif inst[0] == VSUB:
                            vd[0][i * 8 : (i + 1) * 8] = scalar1 - scalar2
                        elif inst[0] == VMUL:
                            vd[0][i * 8 : (i + 1) * 8] = scalar1 * scalar2
                elif ele_type[0] == EEW16:
                    for i in allo.grid(VLEN // 16, name="arith_16"):
                        scalar1: int16 = operand1[i * 16 : (i + 1) * 16]
                        scalar2: int16 = operand2[i * 16 : (i + 1) * 16]
                        if inst[0] == VADD:
                            vd[0][i * 16 : (i + 1) * 16] = scalar1 + scalar2
                        elif inst[0] == VSUB:
                            vd[0][i * 16 : (i + 1) * 16] = scalar1 - scalar2
                        elif inst[0] == VMUL:
                            vd[0][i * 16 : (i + 1) * 16] = scalar1 * scalar2
                elif ele_type[0] == EEW32:
                    for i in allo.grid(VLEN // 32, name="arith_32"):
                        scalar1: int32 = operand1[i * 32 : (i + 1) * 32]
                        scalar2: int32 = operand2[i * 32 : (i + 1) * 32]
                        if inst[0] == VADD:
                            vd[0][i * 32 : (i + 1) * 32] = scalar1 + scalar2
                        elif inst[0] == VSUB:
                            vd[0][i * 32 : (i + 1) * 32] = scalar1 - scalar2
                        elif inst[0] == VMUL:
                            vd[0][i * 32 : (i + 1) * 32] = scalar1 * scalar2

    A = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    B = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    C = np.zeros(VLEN // ELEN).astype(np.uint32)
    packed_A = np.ascontiguousarray(A).view(np_256)
    packed_B = np.ascontiguousarray(B).view(np_256)
    packed_C = np.ascontiguousarray(C).view(np_256)
    scalar = np.array([5], dtype=np.uint32)
    inst = np.array([VMIN], dtype=np.uint8)
    ele_type = np.array([EEW32], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod = df.build(top, target="simulator")
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(np.minimum(A, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VECTOR TEST!")

    s = df.customize(top)
    if hls.is_available("vitis_hls"):
        print("Starting Test...")
        mod = s.build(
            target="vitis_hls",
            mode="hw",
            project=f"vec_hw.prj",
            wrap_io=False,
        )
        mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
        unpacked_C = packed_C.view(np.uint32)
        print(np.minimum(A, B))
        print(unpacked_C)
        np.testing.assert_allclose(np.minimum(A, B), unpacked_C, rtol=1e-5, atol=1e-5)
        print("Passed hw Test!")


if __name__ == "__main__":
    test_vadd()
    test_vec()
