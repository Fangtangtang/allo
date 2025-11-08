# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import uint256, uint32, uint8, bool, int8, int16, int32, Stream
import allo.dataflow as df
from allo.utils import get_np_struct_type
from allo.backend import hls

VLEN = 256
ELEN = 32

import numpy as np

np_256 = get_np_struct_type(VLEN)

# [NOTE] reference: https://github.com/riscvarchive/riscv-v-spec/releases/tag/v1.0

# VLEN: The number of bits in a single vector register
VLEN = 256
# ELEN: The number of bits in an element within a vector register
ELEN = 32

# EEW: effective element width, the size of all the elements within a vector register
EEW8 = 0  # 8bits
EEW16 = 1  # 16bits
EEW32 = 2  # 32bits


# ############################
# Vector Instructions
# ############################
# - min/max
#   vmax.vv: vector-vector element-wise signed maximum
#   vmax.vx: vector-scalar element-wise signed maximum
VMAX = 0
#   vmin.vv: vector-vector element-wise signed minimum
#   vmin.vx: vector-scalar element-wise signed minimum
VMIN = 1

# - add/sub
#   vadd.vv: vector-vector element-wise add
#   vadd.vx: vector-scalar element-wise add
VADD = 2
#   vsub.vv: vector-vector element-wise subtract
#   vsub.vx: vector-scalar element-wise subtract vd[i] = vs2[i] - rs1
VSUB = 3

# - mul
#   vmul.vv: vector-vector element-wise multiply
#   vmul.vx: vector-scalar element-wise multiply
VMUL = 4

# - reduction
#   vredmax.vs: vd = max( vs1 , vs2[*] )
VREDMAX = 5
#   vredsum.vs: vd = sum( vs1 , vs2[*] )
VREDSUM = 6
# ############################


def test_vec():
    INST_NUM = 4

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def VEC(
            inst_: uint8[INST_NUM],  # instruction
            ele_type_: uint8[INST_NUM],  # EEW: effective element width
            vv_: bool[INST_NUM],  # vector-vector (True) or vector-scalar (False)
            vs1_: uint256[INST_NUM],  # input vector register
            vs2_: uint256[INST_NUM],  # input vector register
            rs1_: uint32[INST_NUM],  # input scalar
            vd_: uint256[INST_NUM],  # output vector register
        ):
            for idx in allo.grid(INST_NUM, name="inst_outer"):
                ele_type: uint8 = ele_type_[idx]
                vv: bool = vv_[idx]
                vs1: uint256 = vs1_[idx]
                rs1: uint32 = rs1_[idx]

                # prepare operand
                operand1_8b: uint256
                for i in allo.grid(VLEN // 8, name="scalar_to_vector_8"):
                    operand1_8b[i * 8 : (i + 1) * 8] = rs1[0:8]
                operand1_16b: uint256
                for i in allo.grid(VLEN // 16, name="scalar_to_vector_16"):
                    operand1_16b[i * 16 : (i + 1) * 16] = rs1[0:16]
                operand1_32b: uint256
                for i in allo.grid(VLEN // 32, name="scalar_to_vector_32"):
                    operand1_32b[i * 32 : (i + 1) * 32] = rs1

                operand1: uint256
                operand2: uint256 = vs2_[idx]
                if vv:
                    operand1 = vs1
                else:
                    if ele_type == EEW8:
                        operand1 = operand1_8b
                    elif ele_type == EEW16:
                        operand1 = operand1_16b
                    elif ele_type == EEW32:
                        operand1 = operand1_32b

                # compute

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
                inst: uint8 = inst_[idx]
                vd: uint256
                # write back
                if inst == VADD:
                    if ele_type == EEW8:
                        vd = add_8b
                    if ele_type == EEW16:
                        vd = add_16b
                    if ele_type == EEW32:
                        vd = add_32b
                if inst == VSUB:
                    if ele_type == EEW8:
                        vd = sub_8b
                    if ele_type == EEW16:
                        vd = sub_16b
                    if ele_type == EEW32:
                        vd = sub_32b
                if inst == VMUL:
                    if ele_type == EEW8:
                        vd = mul_8b
                    if ele_type == EEW16:
                        vd = mul_16b
                    if ele_type == EEW32:
                        vd = mul_32b

                vd_[idx] = vd

    A = np.random.randint(0, 64, (INST_NUM, VLEN // ELEN)).astype(np.uint32)
    B = np.random.randint(0, 64, (INST_NUM, VLEN // ELEN)).astype(np.uint32)
    C = np.zeros((INST_NUM, VLEN // ELEN)).astype(np.uint32)
    packed_A = np.ascontiguousarray(A).view(np_256).reshape((INST_NUM,))
    packed_B = np.ascontiguousarray(B).view(np_256).reshape((INST_NUM,))
    packed_C = np.ascontiguousarray(C).view(np_256).reshape((INST_NUM,))
    scalar = np.array([5] * INST_NUM, dtype=np.uint32)
    ele_type = np.array([EEW32] * INST_NUM, dtype=np.uint8)
    mod = df.build(top, target="simulator")

    s = df.customize(top)

    s.pipeline(s.get_loops("VEC_0")["inst_outer"]["idx"])

    if hls.is_available("vitis_hls"):
        print("Starting Test...")
        mod = s.build(
            target="vitis_hls",
            mode="hw_emu",
            project=f"vec_hw_emu.prj",
            wrap_io=False,
        )
        inst = np.array([VMUL] * INST_NUM, dtype=np.uint8)
        vv = np.array([True] * INST_NUM, dtype=np.bool_)
        mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
        unpacked_C = packed_C.view(np.uint32).reshape(INST_NUM, VLEN // ELEN)
        print(np.multiply(A, B))
        print(unpacked_C)
        np.testing.assert_allclose(np.multiply(A, B), unpacked_C, rtol=1e-5, atol=1e-5)
        print("Passed hw Test!")


if __name__ == "__main__":
    test_vec()
