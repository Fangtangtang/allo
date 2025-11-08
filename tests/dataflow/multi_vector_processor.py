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
    INST_LEN = 1

    @df.region()
    def top():
        inst_: Stream[uint8, 1]
        ele_type_: Stream[uint8, 1]
        vv_: Stream[bool, 1]
        vs1_: Stream[uint256, 1]
        vs2_: Stream[uint256, 1]
        rs1_: Stream[uint32, 1]

        vd_: Stream[uint256, 1]

        @df.kernel(mapping=[1])
        def DECODE(
            inst_list: uint8[INST_LEN],  # instruction
            ele_type_list: uint8[INST_LEN],  # EEW: effective element width
            vv_list: bool[INST_LEN],  # vector-vector (True) or vector-scalar (False)
            vs1_list: uint256[INST_LEN],  # input vector register
            vs2_list: uint256[INST_LEN],  # input vector register
            rs1_list: uint32[INST_LEN],  # input scalar
        ):
            for i in allo.grid(INST_LEN, name="inst_loop"):
                inst_.put(inst_list[i])
                ele_type_.put(ele_type_list[i])
                vv_.put(vv_list[i])
                vs1_.put(vs1_list[i])
                vs2_.put(vs2_list[i])
                rs1_.put(rs1_list[i])

        @df.kernel(mapping=[1])
        def VEC():
            ele_type: uint8 = ele_type_.get()
            vv: bool = vv_.get()
            vs1: uint256 = vs1_.get()
            vs2: uint256 = vs2_.get()
            rs1: uint32 = rs1_.get()

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
            operand2: uint256 = vs2
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
            # reduction max
            # - 8 bits
            base_8b: int8 = operand1[0:8]
            operand1_8b_0: int8 = operand2[0:8]
            operand1_8b_1: int8 = operand2[8:16]
            operand1_8b_2: int8 = operand2[16:24]
            operand1_8b_3: int8 = operand2[24:32]
            operand1_8b_4: int8 = operand2[32:40]
            operand1_8b_5: int8 = operand2[40:48]
            operand1_8b_6: int8 = operand2[48:56]
            operand1_8b_7: int8 = operand2[56:64]
            operand1_8b_8: int8 = operand2[64:72]
            operand1_8b_9: int8 = operand2[72:80]
            operand1_8b_10: int8 = operand2[80:88]
            operand1_8b_11: int8 = operand2[88:96]
            operand1_8b_12: int8 = operand2[96:104]
            operand1_8b_13: int8 = operand2[104:112]
            operand1_8b_14: int8 = operand2[112:120]
            operand1_8b_15: int8 = operand2[120:128]
            operand1_8b_16: int8 = operand2[128:136]
            operand1_8b_17: int8 = operand2[136:144]
            operand1_8b_18: int8 = operand2[144:152]
            operand1_8b_19: int8 = operand2[152:160]
            operand1_8b_20: int8 = operand2[160:168]
            operand1_8b_21: int8 = operand2[168:176]
            operand1_8b_22: int8 = operand2[176:184]
            operand1_8b_23: int8 = operand2[184:192]
            operand1_8b_24: int8 = operand2[192:200]
            operand1_8b_25: int8 = operand2[200:208]
            operand1_8b_26: int8 = operand2[208:216]
            operand1_8b_27: int8 = operand2[216:224]
            operand1_8b_28: int8 = operand2[224:232]
            operand1_8b_29: int8 = operand2[232:240]
            operand1_8b_30: int8 = operand2[240:248]
            operand1_8b_31: int8 = operand2[248:256]

            redmax_8b_l0_0: int8
            if operand1_8b_0 > operand1_8b_1:
                redmax_8b_l0_0 = operand1_8b_0
            else:
                redmax_8b_l0_0 = operand1_8b_1
            redmax_8b_l0_1: int8
            if operand1_8b_2 > operand1_8b_3:
                redmax_8b_l0_1 = operand1_8b_2
            else:
                redmax_8b_l0_1 = operand1_8b_3
            redmax_8b_l0_2: int8
            if operand1_8b_4 > operand1_8b_5:
                redmax_8b_l0_2 = operand1_8b_4
            else:
                redmax_8b_l0_2 = operand1_8b_5
            redmax_8b_l0_3: int8
            if operand1_8b_6 > operand1_8b_7:
                redmax_8b_l0_3 = operand1_8b_6
            else:
                redmax_8b_l0_3 = operand1_8b_7
            redmax_8b_l0_4: int8
            if operand1_8b_8 > operand1_8b_9:
                redmax_8b_l0_4 = operand1_8b_8
            else:
                redmax_8b_l0_4 = operand1_8b_9
            redmax_8b_l0_5: int8
            if operand1_8b_10 > operand1_8b_11:
                redmax_8b_l0_5 = operand1_8b_10
            else:
                redmax_8b_l0_5 = operand1_8b_11
            redmax_8b_l0_6: int8
            if operand1_8b_12 > operand1_8b_13:
                redmax_8b_l0_6 = operand1_8b_12
            else:
                redmax_8b_l0_6 = operand1_8b_13
            redmax_8b_l0_7: int8
            if operand1_8b_14 > operand1_8b_15:
                redmax_8b_l0_7 = operand1_8b_14
            else:
                redmax_8b_l0_7 = operand1_8b_15
            redmax_8b_l0_8: int8
            if operand1_8b_16 > operand1_8b_17:
                redmax_8b_l0_8 = operand1_8b_16
            else:
                redmax_8b_l0_8 = operand1_8b_17
            redmax_8b_l0_9: int8
            if operand1_8b_18 > operand1_8b_19:
                redmax_8b_l0_9 = operand1_8b_18
            else:
                redmax_8b_l0_9 = operand1_8b_19
            redmax_8b_l0_10: int8
            if operand1_8b_20 > operand1_8b_21:
                redmax_8b_l0_10 = operand1_8b_20
            else:
                redmax_8b_l0_10 = operand1_8b_21
            redmax_8b_l0_11: int8
            if operand1_8b_22 > operand1_8b_23:
                redmax_8b_l0_11 = operand1_8b_22
            else:
                redmax_8b_l0_11 = operand1_8b_23
            redmax_8b_l0_12: int8
            if operand1_8b_24 > operand1_8b_25:
                redmax_8b_l0_12 = operand1_8b_24
            else:
                redmax_8b_l0_12 = operand1_8b_25
            redmax_8b_l0_13: int8
            if operand1_8b_26 > operand1_8b_27:
                redmax_8b_l0_13 = operand1_8b_26
            else:
                redmax_8b_l0_13 = operand1_8b_27
            redmax_8b_l0_14: int8
            if operand1_8b_28 > operand1_8b_29:
                redmax_8b_l0_14 = operand1_8b_28
            else:
                redmax_8b_l0_14 = operand1_8b_29
            redmax_8b_l0_15: int8
            if operand1_8b_30 > operand1_8b_31:
                redmax_8b_l0_15 = operand1_8b_30
            else:
                redmax_8b_l0_15 = operand1_8b_31

            redmax_8b_l1_0: int8
            if redmax_8b_l0_0 > redmax_8b_l0_1:
                redmax_8b_l1_0 = redmax_8b_l0_0
            else:
                redmax_8b_l1_0 = redmax_8b_l0_1
            redmax_8b_l1_1: int8
            if redmax_8b_l0_2 > redmax_8b_l0_3:
                redmax_8b_l1_1 = redmax_8b_l0_2
            else:
                redmax_8b_l1_1 = redmax_8b_l0_3
            redmax_8b_l1_2: int8
            if redmax_8b_l0_4 > redmax_8b_l0_5:
                redmax_8b_l1_2 = redmax_8b_l0_4
            else:
                redmax_8b_l1_2 = redmax_8b_l0_5
            redmax_8b_l1_3: int8
            if redmax_8b_l0_6 > redmax_8b_l0_7:
                redmax_8b_l1_3 = redmax_8b_l0_6
            else:
                redmax_8b_l1_3 = redmax_8b_l0_7
            redmax_8b_l1_4: int8
            if redmax_8b_l0_8 > redmax_8b_l0_9:
                redmax_8b_l1_4 = redmax_8b_l0_8
            else:
                redmax_8b_l1_4 = redmax_8b_l0_9
            redmax_8b_l1_5: int8
            if redmax_8b_l0_10 > redmax_8b_l0_11:
                redmax_8b_l1_5 = redmax_8b_l0_10
            else:
                redmax_8b_l1_5 = redmax_8b_l0_11
            redmax_8b_l1_6: int8
            if redmax_8b_l0_12 > redmax_8b_l0_13:
                redmax_8b_l1_6 = redmax_8b_l0_12
            else:
                redmax_8b_l1_6 = redmax_8b_l0_13
            redmax_8b_l1_7: int8
            if redmax_8b_l0_14 > redmax_8b_l0_15:
                redmax_8b_l1_7 = redmax_8b_l0_14
            else:
                redmax_8b_l1_7 = redmax_8b_l0_15

            redmax_8b_l2_0: int8
            if redmax_8b_l1_0 > redmax_8b_l1_1:
                redmax_8b_l2_0 = redmax_8b_l1_0
            else:
                redmax_8b_l2_0 = redmax_8b_l1_1
            redmax_8b_l2_1: int8
            if redmax_8b_l1_2 > redmax_8b_l1_3:
                redmax_8b_l2_1 = redmax_8b_l1_2
            else:
                redmax_8b_l2_1 = redmax_8b_l1_3
            redmax_8b_l2_2: int8
            if redmax_8b_l1_4 > redmax_8b_l1_5:
                redmax_8b_l2_2 = redmax_8b_l1_4
            else:
                redmax_8b_l2_2 = redmax_8b_l1_5
            redmax_8b_l2_3: int8
            if redmax_8b_l1_6 > redmax_8b_l1_7:
                redmax_8b_l2_3 = redmax_8b_l1_6
            else:
                redmax_8b_l2_3 = redmax_8b_l1_7

            redmax_8b_l3_0: int8
            if redmax_8b_l2_0 > redmax_8b_l2_1:
                redmax_8b_l3_0 = redmax_8b_l2_0
            else:
                redmax_8b_l3_0 = redmax_8b_l2_1
            redmax_8b_l3_1: int8
            if redmax_8b_l2_2 > redmax_8b_l2_3:
                redmax_8b_l3_1 = redmax_8b_l2_2
            else:
                redmax_8b_l3_1 = redmax_8b_l2_3

            redmax_8b_l4_0: int8
            if redmax_8b_l3_0 > redmax_8b_l3_1:
                redmax_8b_l4_0 = redmax_8b_l3_0
            else:
                redmax_8b_l4_0 = redmax_8b_l3_1
            redmax_8b: int8
            if base_8b > redmax_8b_l4_0:
                redmax_8b = base_8b
            else:
                redmax_8b = redmax_8b_l4_0

            # - 16 bits
            base_16b: int16 = operand1[0:16]
            operand_16b_0: int16 = operand2[0:16]
            operand_16b_1: int16 = operand2[16:32]
            operand_16b_2: int16 = operand2[32:48]
            operand_16b_3: int16 = operand2[48:64]
            operand_16b_4: int16 = operand2[64:80]
            operand_16b_5: int16 = operand2[80:96]
            operand_16b_6: int16 = operand2[96:112]
            operand_16b_7: int16 = operand2[112:128]
            operand_16b_8: int16 = operand2[128:144]
            operand_16b_9: int16 = operand2[144:160]
            operand_16b_10: int16 = operand2[160:176]
            operand_16b_11: int16 = operand2[176:192]
            operand_16b_12: int16 = operand2[192:208]
            operand_16b_13: int16 = operand2[208:224]
            operand_16b_14: int16 = operand2[224:240]
            operand_16b_15: int16 = operand2[240:256]

            redmax_16b_l0_0: int16
            if operand_16b_0 > operand_16b_1:
                redmax_16b_l0_0 = operand_16b_0
            else:
                redmax_16b_l0_0 = operand_16b_1
            redmax_16b_l0_1: int16
            if operand_16b_2 > operand_16b_3:
                redmax_16b_l0_1 = operand_16b_2
            else:
                redmax_16b_l0_1 = operand_16b_3
            redmax_16b_l0_2: int16
            if operand_16b_4 > operand_16b_5:
                redmax_16b_l0_2 = operand_16b_4
            else:
                redmax_16b_l0_2 = operand_16b_5
            redmax_16b_l0_3: int16
            if operand_16b_6 > operand_16b_7:
                redmax_16b_l0_3 = operand_16b_6
            else:
                redmax_16b_l0_3 = operand_16b_7
            redmax_16b_l0_4: int16
            if operand_16b_8 > operand_16b_9:
                redmax_16b_l0_4 = operand_16b_8
            else:
                redmax_16b_l0_4 = operand_16b_9
            redmax_16b_l0_5: int16
            if operand_16b_10 > operand_16b_11:
                redmax_16b_l0_5 = operand_16b_10
            else:
                redmax_16b_l0_5 = operand_16b_11
            redmax_16b_l0_6: int16
            if operand_16b_12 > operand_16b_13:
                redmax_16b_l0_6 = operand_16b_12
            else:
                redmax_16b_l0_6 = operand_16b_13
            redmax_16b_l0_7: int16
            if operand_16b_14 > operand_16b_15:
                redmax_16b_l0_7 = operand_16b_14
            else:
                redmax_16b_l0_7 = operand_16b_15

            redmax_16b_l1_0: int16
            if redmax_16b_l0_0 > redmax_16b_l0_1:
                redmax_16b_l1_0 = redmax_16b_l0_0
            else:
                redmax_16b_l1_0 = redmax_16b_l0_1
            redmax_16b_l1_1: int16
            if redmax_16b_l0_2 > redmax_16b_l0_3:
                redmax_16b_l1_1 = redmax_16b_l0_2
            else:
                redmax_16b_l1_1 = redmax_16b_l0_3
            redmax_16b_l1_2: int16
            if redmax_16b_l0_4 > redmax_16b_l0_5:
                redmax_16b_l1_2 = redmax_16b_l0_4
            else:
                redmax_16b_l1_2 = redmax_16b_l0_5
            redmax_16b_l1_3: int16
            if redmax_16b_l0_6 > redmax_16b_l0_7:
                redmax_16b_l1_3 = redmax_16b_l0_6
            else:
                redmax_16b_l1_3 = redmax_16b_l0_7

            redmax_16b_l2_0: int16
            if redmax_16b_l1_0 > redmax_16b_l1_1:
                redmax_16b_l2_0 = redmax_16b_l1_0
            else:
                redmax_16b_l2_0 = redmax_16b_l1_1
            redmax_16b_l2_1: int16
            if redmax_16b_l1_2 > redmax_16b_l1_3:
                redmax_16b_l2_1 = redmax_16b_l1_2
            else:
                redmax_16b_l2_1 = redmax_16b_l1_3

            redmax_16b_l3_0: int16
            if redmax_16b_l2_0 > redmax_16b_l2_1:
                redmax_16b_l3_0 = redmax_16b_l2_0
            else:
                redmax_16b_l3_0 = redmax_16b_l2_1

            redmax_16b: int16
            if base_16b > redmax_16b_l3_0:
                redmax_16b = base_16b
            else:
                redmax_16b = redmax_16b_l3_0

            # # - 32 bits
            base_32b: int32 = operand1[0:32]
            operand_32b_0: int32 = operand2[0:32]
            operand_32b_1: int32 = operand2[32:64]
            operand_32b_2: int32 = operand2[64:96]
            operand_32b_3: int32 = operand2[96:128]
            operand_32b_4: int32 = operand2[128:160]
            operand_32b_5: int32 = operand2[160:192]
            operand_32b_6: int32 = operand2[192:224]
            operand_32b_7: int32 = operand2[224:256]

            redmax_32b_l0_0: int32
            if operand_32b_0 > operand_32b_1:
                redmax_32b_l0_0 = operand_32b_0
            else:
                redmax_32b_l0_0 = operand_32b_1
            redmax_32b_l0_1: int32
            if operand_32b_2 > operand_32b_3:
                redmax_32b_l0_1 = operand_32b_2
            else:
                redmax_32b_l0_1 = operand_32b_3
            redmax_32b_l0_2: int32
            if operand_32b_4 > operand_32b_5:
                redmax_32b_l0_2 = operand_32b_4
            else:
                redmax_32b_l0_2 = operand_32b_5
            redmax_32b_l0_3: int32
            if operand_32b_6 > operand_32b_7:
                redmax_32b_l0_3 = operand_32b_6
            else:
                redmax_32b_l0_3 = operand_32b_7
            redmax_32b_l1_0: int32
            if redmax_32b_l0_0 > redmax_32b_l0_1:
                redmax_32b_l1_0 = redmax_32b_l0_0
            else:
                redmax_32b_l1_0 = redmax_32b_l0_1
            redmax_32b_l1_1: int32
            if redmax_32b_l0_2 > redmax_32b_l0_3:
                redmax_32b_l1_1 = redmax_32b_l0_2
            else:
                redmax_32b_l1_1 = redmax_32b_l0_3

            redmax_32b_l2_0: int32 = operand1[0:32]
            if redmax_32b_l1_0 > redmax_32b_l1_1:
                redmax_32b_l2_0 = redmax_32b_l1_0
            else:
                redmax_32b_l2_0 = redmax_32b_l1_1
            redmax_32b: int32
            if base_32b > redmax_32b_l2_0:
                redmax_32b = base_32b
            else:
                redmax_32b = redmax_32b_l2_0

            redsum_8b: int8 = operand1[0:8]
            redsum_16b: int16 = operand1[0:16]
            redsum_32b: int32 = operand1[0:32]

            for i in allo.grid(VLEN // 8, name="reduction_8"):
                scalar: int8 = operand2[i * 8 : (i + 1) * 8]
                redsum_8b = redsum_8b + scalar
            for i in allo.grid(VLEN // 16, name="reduction_16"):
                scalar: int16 = operand2[i * 16 : (i + 1) * 16]
                redsum_16b = redsum_16b + scalar
            for i in allo.grid(VLEN // 32, name="reduction_32"):
                scalar: int32 = operand2[i * 32 : (i + 1) * 32]
                redsum_32b = redsum_32b + scalar

            inst: uint8 = inst_.get()
            vd: uint256
            # write back
            if inst == VMAX:
                if ele_type == EEW8:
                    vd = max_8b
                elif ele_type == EEW16:
                    vd = max_16b
                elif ele_type == EEW32:
                    vd = max_32b
            elif inst == VMIN:
                if ele_type == EEW8:
                    vd = min_8b
                elif ele_type == EEW16:
                    vd = min_16b
                elif ele_type == EEW32:
                    vd = min_32b
            elif inst == VADD:
                if ele_type == EEW8:
                    vd = add_8b
                elif ele_type == EEW16:
                    vd = add_16b
                elif ele_type == EEW32:
                    vd = add_32b
            elif inst == VSUB:
                if ele_type == EEW8:
                    vd = sub_8b
                elif ele_type == EEW16:
                    vd = sub_16b
                elif ele_type == EEW32:
                    vd = sub_32b
            elif inst == VMUL:
                if ele_type == EEW8:
                    vd = mul_8b
                elif ele_type == EEW16:
                    vd = mul_16b
                elif ele_type == EEW32:
                    vd = mul_32b
            elif inst == VREDMAX:
                if ele_type == EEW8:
                    vd[0:8] = redmax_8b
                elif ele_type == EEW16:
                    vd[0:16] = redmax_16b
                if ele_type == EEW32:
                    vd[0:32] = redmax_32b
            elif inst == VREDSUM:
                if ele_type == EEW8:
                    vd[0:8] = redsum_8b
                elif ele_type == EEW16:
                    vd[0:16] = redsum_16b
                elif ele_type == EEW32:
                    vd[0:32] = redsum_32b
            vd_.put(vd)

        @df.kernel(mapping=[1])
        def OUTPUT(
            vd_list: uint256[INST_LEN],  # output vector register
        ):
            for i in allo.grid(INST_LEN, name="output_loop"):
                vd_list[i] = vd_.get()

    A = np.random.randint(0, 64, (INST_LEN, VLEN // ELEN)).astype(np.uint32)
    B = np.random.randint(0, 64, (INST_LEN, VLEN // ELEN)).astype(np.uint32)
    C = np.zeros((INST_LEN, VLEN // ELEN)).astype(np.uint32)
    packed_A = np.ascontiguousarray(A).view(np_256).reshape((INST_LEN,))
    packed_B = np.ascontiguousarray(B).view(np_256).reshape((INST_LEN,))
    packed_C = np.ascontiguousarray(C).view(np_256).reshape((INST_LEN,))
    scalar = np.array([5], dtype=np.uint32)
    ele_type = np.array([EEW32], dtype=np.uint8)
    mod = df.build(top, target="simulator")

    # MIN
    inst = np.array([VMIN], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32).reshape((INST_LEN, VLEN // ELEN))
    np.testing.assert_allclose(np.minimum(A, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VV MIN TEST!")
    vv = np.array([False], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32).reshape((INST_LEN, VLEN // ELEN))
    np.testing.assert_allclose(np.minimum(scalar, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VX MIN TEST!")

    # MAX
    inst = np.array([VMAX], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32).reshape((INST_LEN, VLEN // ELEN))
    np.testing.assert_allclose(np.maximum(A, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VV MAX TEST!")
    vv = np.array([False], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32).reshape((INST_LEN, VLEN // ELEN))
    np.testing.assert_allclose(np.maximum(scalar, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VX MAX TEST!")

    # ADD
    inst = np.array([VADD], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32).reshape((INST_LEN, VLEN // ELEN))
    np.testing.assert_allclose(np.add(A, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VV ADD TEST!")
    vv = np.array([False], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32).reshape((INST_LEN, VLEN // ELEN))
    np.testing.assert_allclose(np.add(scalar, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VX ADD TEST!")

    # SUB
    inst = np.array([VSUB], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32).reshape((INST_LEN, VLEN // ELEN))
    np.testing.assert_allclose(np.subtract(B, A), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VV SUB TEST!")
    vv = np.array([False], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32).reshape((INST_LEN, VLEN // ELEN))
    np.testing.assert_allclose(np.subtract(B, scalar), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VX SUB TEST!")

    # MUL
    inst = np.array([VMUL], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32).reshape((INST_LEN, VLEN // ELEN))
    np.testing.assert_allclose(np.multiply(A, B), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VV MUL TEST!")
    vv = np.array([False], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32).reshape((INST_LEN, VLEN // ELEN))
    np.testing.assert_allclose(np.multiply(B, scalar), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VX MUL TEST!")

    init = np.zeros((INST_LEN, VLEN // ELEN)).astype(np.uint32)
    packed_init = np.ascontiguousarray(init).view(np_256).reshape((INST_LEN,))

    # REDMAX
    inst = np.array([VREDMAX], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_init, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32).reshape((INST_LEN, VLEN // ELEN))[:, 0]
    np.testing.assert_allclose(np.max(B, axis=1), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VV REDMAX TEST!")

    # REDSUM
    inst = np.array([VREDSUM], dtype=np.uint8)
    vv = np.array([True], dtype=np.bool_)
    mod(inst, ele_type, vv, packed_init, packed_B, scalar, packed_C)
    unpacked_C = packed_C.view(np.uint32).reshape((INST_LEN, VLEN // ELEN))[:, 0]
    np.testing.assert_allclose(np.sum(B, axis=1), unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED VV REDSUM TEST!")

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

    s.pipeline(s.get_loops("VEC_0")["reduction_8"]["i"])
    s.pipeline(s.get_loops("VEC_0")["reduction_16"]["i"])
    s.pipeline(s.get_loops("VEC_0")["reduction_32"]["i"])

    if hls.is_available("vitis_hls"):
        print("Starting Test...")
        mod = s.build(
            target="vitis_hls",
            mode="hw_emu",
            project=f"vec_hw_emu.prj",
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
    test_vec()
