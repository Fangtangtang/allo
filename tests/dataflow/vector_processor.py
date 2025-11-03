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

# - reduction
VREDMAX = 5  # vd[0] = maxu( vs1[0] , vs2[*] )
VREDSUM = 6  # vd[0] = sum( vs1[0] , vs2[*] )
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
            operand1_8b: uint256
            for i in allo.grid(VLEN // 8, name="scalar_to_vector_8"):
                operand1_8b[i * 8 : (i + 1) * 8] = rs1[0][0:8]
            operand1_16b: uint256
            for i in allo.grid(VLEN // 16, name="scalar_to_vector_16"):
                operand1_16b[i * 16 : (i + 1) * 16] = rs1[0][0:16]
            operand1_32b: uint256
            for i in allo.grid(VLEN // 32, name="scalar_to_vector_32"):
                operand1_32b[i * 32 : (i + 1) * 32] = rs1[0]

            operand1: uint256
            operand2: uint256 = vs2[0]
            if vv[0]:
                operand1 = vs1[0]
            else:
                if ele_type[0] == EEW8:
                    operand1 = operand1_8b
                elif ele_type[0] == EEW16:
                    operand1 = operand1_16b
                elif ele_type[0] == EEW32:
                    operand1 = operand1_32b

            # compute
            # - min/max
            min_8b: uint256
            min_16b: uint256
            min_32b: uint256
            max_8b: uint256
            max_16b: uint256
            max_32b: uint256
            for i in allo.grid(VLEN // 8, name="min_max_8"):
                scalar1: int8 = operand1[i * 8 : (i + 1) * 8]
                scalar2: int8 = operand2[i * 8 : (i + 1) * 8]
                condition: bool = scalar1 > scalar2
                if condition:
                    min_8b[i * 8 : (i + 1) * 8] = scalar2
                    max_8b[i * 8 : (i + 1) * 8] = scalar1
                else:
                    max_8b[i * 8 : (i + 1) * 8] = scalar2
                    min_8b[i * 8 : (i + 1) * 8] = scalar1
            for i in allo.grid(VLEN // 16, name="min_max_16"):
                scalar1: int16 = operand1[i * 16 : (i + 1) * 16]
                scalar2: int16 = operand2[i * 16 : (i + 1) * 16]
                condition: bool = scalar1 > scalar2
                if condition:
                    min_16b[i * 16 : (i + 1) * 16] = scalar2
                    max_16b[i * 16 : (i + 1) * 16] = scalar1
                else:
                    max_16b[i * 16 : (i + 1) * 16] = scalar2
                    min_16b[i * 16 : (i + 1) * 16] = scalar1
            for i in allo.grid(VLEN // 32, name="min_max_32"):
                scalar1: int32 = operand1[i * 32 : (i + 1) * 32]
                scalar2: int32 = operand2[i * 32 : (i + 1) * 32]
                condition: bool = scalar1 > scalar2
                if condition:
                    min_32b[i * 32 : (i + 1) * 32] = scalar2
                    max_32b[i * 32 : (i + 1) * 32] = scalar1
                else:
                    max_32b[i * 32 : (i + 1) * 32] = scalar2
                    min_32b[i * 32 : (i + 1) * 32] = scalar1

            # - add/sub/mul
            add_8b: uint256
            add_16b: uint256
            add_32b: uint256
            sub_8b: uint256
            sub_16b: uint256
            sub_32b: uint256
            mul_8b: uint256
            mul_16b: uint256
            mul_32b: uint256
            for i in allo.grid(VLEN // 8, name="arith_8"):
                scalar1: int8 = operand1[i * 8 : (i + 1) * 8]
                scalar2: int8 = operand2[i * 8 : (i + 1) * 8]
                add_8b[i * 8 : (i + 1) * 8] = scalar2 + scalar1
                sub_8b[i * 8 : (i + 1) * 8] = scalar2 - scalar1
                mul_8b[i * 8 : (i + 1) * 8] = scalar2 * scalar1
            for i in allo.grid(VLEN // 16, name="arith_16"):
                scalar1: int16 = operand1[i * 16 : (i + 1) * 16]
                scalar2: int16 = operand2[i * 16 : (i + 1) * 16]
                add_16b[i * 16 : (i + 1) * 16] = scalar2 + scalar1
                sub_16b[i * 16 : (i + 1) * 16] = scalar2 - scalar1
                mul_16b[i * 16 : (i + 1) * 16] = scalar2 * scalar1
            for i in allo.grid(VLEN // 32, name="arith_32"):
                scalar1: int32 = operand1[i * 32 : (i + 1) * 32]
                scalar2: int32 = operand2[i * 32 : (i + 1) * 32]
                add_32b[i * 32 : (i + 1) * 32] = scalar2 + scalar1
                sub_32b[i * 32 : (i + 1) * 32] = scalar2 - scalar1
                mul_32b[i * 32 : (i + 1) * 32] = scalar2 * scalar1

            # reduction
            redmax_8b: int8 = operand1[0:8]
            redmax_16b: int16 = operand1[0:16]
            redmax_32b: int32 = operand1[0:32]
            redsum_8b: int8 = operand1[0:8]
            redsum_16b: int16 = operand1[0:16]
            redsum_32b: int32 = operand1[0:32]
            for i in allo.grid(VLEN // 8, name="reduction_8"):
                scalar: int8 = operand2[i * 8 : (i + 1) * 8]
                if scalar > redmax_8b:
                    redmax_8b = scalar
                redsum_8b = redsum_8b + scalar
            for i in allo.grid(VLEN // 16, name="reduction_16"):
                scalar: int16 = operand2[i * 16 : (i + 1) * 16]
                if scalar > redmax_16b:
                    redmax_16b = scalar
                redsum_16b = redsum_16b + scalar
            for i in allo.grid(VLEN // 32, name="reduction_32"):
                scalar: int32 = operand2[i * 32 : (i + 1) * 32]
                if scalar > redmax_32b:
                    redmax_32b = scalar
                redsum_32b = redsum_32b + scalar

            # write back
            if inst[0] == VMAX:
                if ele_type[0] == EEW8:
                    vd[0] = max_8b
                elif ele_type[0] == EEW16:
                    vd[0] = max_16b
                elif ele_type[0] == EEW32:
                    vd[0] = max_32b
            elif inst[0] == VMIN:
                if ele_type[0] == EEW8:
                    vd[0] = min_8b
                elif ele_type[0] == EEW16:
                    vd[0] = min_16b
                elif ele_type[0] == EEW32:
                    vd[0] = min_32b
            elif inst[0] == VADD:
                if ele_type[0] == EEW8:
                    vd[0] = add_8b
                elif ele_type[0] == EEW16:
                    vd[0] = add_16b
                elif ele_type[0] == EEW32:
                    vd[0] = add_32b
            elif inst[0] == VSUB:
                if ele_type[0] == EEW8:
                    vd[0] = sub_8b
                elif ele_type[0] == EEW16:
                    vd[0] = sub_16b
                elif ele_type[0] == EEW32:
                    vd[0] = sub_32b
            elif inst[0] == VMUL:
                if ele_type[0] == EEW8:
                    vd[0] = mul_8b
                elif ele_type[0] == EEW16:
                    vd[0] = mul_16b
                elif ele_type[0] == EEW32:
                    vd[0] = mul_32b
            elif inst[0] == VREDMAX:
                if ele_type[0] == EEW8:
                    vd[0][0:8] = redmax_8b
                elif ele_type[0] == EEW16:
                    vd[0][0:16] = redmax_16b
                elif ele_type[0] == EEW32:
                    vd[0][0:32] = redmax_32b
            elif inst[0] == VREDSUM:
                if ele_type[0] == EEW8:
                    vd[0][0:8] = redsum_8b
                elif ele_type[0] == EEW16:
                    vd[0][0:16] = redsum_16b
                elif ele_type[0] == EEW32:
                    vd[0][0:32] = redsum_32b

    A = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    B = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    C = np.zeros(VLEN // ELEN).astype(np.uint32)
    packed_A = np.ascontiguousarray(A).view(np_256)
    packed_B = np.ascontiguousarray(B).view(np_256)
    packed_C = np.ascontiguousarray(C).view(np_256)
    scalar = np.array([5], dtype=np.uint32)
    ele_type = np.array([EEW32], dtype=np.uint8)
    mod = df.build(top, target="simulator")

    # MIN
    inst = np.array([VMIN], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(np.minimum(A, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VV MIN TEST!")
    vv = np.array([False], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(np.minimum(scalar, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VX MIN TEST!")

    # MAX
    inst = np.array([VMAX], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(np.maximum(A, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VV MAX TEST!")
    vv = np.array([False], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(np.maximum(scalar, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VX MAX TEST!")

    # ADD
    inst = np.array([VADD], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(np.add(A, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VV ADD TEST!")
    vv = np.array([False], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(np.add(scalar, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VX ADD TEST!")

    # SUB
    inst = np.array([VSUB], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(np.subtract(B, A), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VV SUB TEST!")
    vv = np.array([False], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(np.subtract(B, scalar), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VX SUB TEST!")

    # MUL
    inst = np.array([VMUL], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(np.multiply(A, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VV MUL TEST!")
    vv = np.array([False], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(np.multiply(B, scalar), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VX MUL TEST!")

    init = np.zeros(VLEN // ELEN).astype(np.uint32)
    packed_init = np.ascontiguousarray(init).view(np_256)

    # REDMAX
    inst = np.array([VREDMAX], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_init, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(np.max(B), unpacked_C[0], rtol=1e-5, atol=1e-5)
    print("PASSED VV REDMAX TEST!")

    # REDSUM
    inst = np.array([VREDSUM], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_init, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(np.sum(B), unpacked_C[0], rtol=1e-5, atol=1e-5)
    print("PASSED VV REDSUM TEST!")

    return

    s = df.customize(top)

    s.unroll(s.get_loops("VEC_0")["scalar_to_vector_8"]["i"])
    s.unroll(s.get_loops("VEC_0")["scalar_to_vector_16"]["i"])
    s.unroll(s.get_loops("VEC_0")["scalar_to_vector_32"]["i"])

    s.unroll(s.get_loops("VEC_0")["min_max_8"]["i"])
    s.unroll(s.get_loops("VEC_0")["min_max_16"]["i"])
    s.unroll(s.get_loops("VEC_0")["min_max_32"]["i"])

    s.unroll(s.get_loops("VEC_0")["arith_8"]["i"])
    s.unroll(s.get_loops("VEC_0")["arith_16"]["i"])
    s.unroll(s.get_loops("VEC_0")["arith_32"]["i"])

    if hls.is_available("vitis_hls"):
        print("Starting Test...")
        mod = s.build(
            target="vitis_hls",
            mode="hw",
            project=f"vec_hw.prj",
            wrap_io=False,
        )
        inst = np.array([VMUL], dtype=np.uint8)
        vv = np.array([True], dtype=np.bool_)
        mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
        unpacked_C = packed_C.view(np.uint32)
        print(np.multiply(A, B))
        print(unpacked_C)
        np.testing.assert_allclose(np.multiply(A, B), unpacked_C, rtol=1e-5, atol=1e-5)
        print("Passed hw Test!")


if __name__ == "__main__":
    # test_vadd()
    test_vec()
