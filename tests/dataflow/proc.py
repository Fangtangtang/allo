# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import uint256, uint32, uint8, bool, int8, int32, UInt, Stream
import allo.dataflow as df
from allo.utils import get_np_struct_type
from allo.backend import hls

# ############################
# Instruction Config
# ############################

# instrution has 32 bits
# +-+----------+----------------+----------+------------+
# |-| addr1 (8) | addr2/imm (8) | addr3 (8) | opcode (4) |
# +-+--------+----------------+----------+-------------+


def encode(rs1, rs2, rd, opcode):
    if not (0 <= rs1 < 256 and 0 <= rs2 < 256 and 0 <= rd < 256 and 0 <= opcode < 16):
        raise ValueError("field out of range")

    value = (rs1 << 20) | (rs2 << 12) | (rd << 4) | (opcode << 0)
    print(bin(value))
    return value

# ---- opcode
#
ALLOC = 0  # starting line: addr1; line cnt: imm
FLUSH = 1  # polling untill all WAIT_VEC/WAIT_TENSOR returned and set to CLEAN
COPY_TO_MEM = 2  # copy by line
COPY_FROM_MEM = 3

VEC_ADD = 6

# -- vector
# [VV/VX][vector op]
VV = 0
VX = 1  # TODO: not supported for now
VMAX = 0
VMIN = 1
VADD = 2
VSUB = 3

# -- tensor
MATMUL = 0

# ############################
# MMU Config
# ############################
WIDTH = 256  # 256 bits
HEIGHT = 32
np_256 = get_np_struct_type(WIDTH)

# ---- bit map config
INVALID = 0  # unallocated
WAIT_VEC = 1  # waiting write back from vec
WAIT_TENSOR = 2  # waiting write back from tensor
CLEAN = 3  # valid data

# ############################
# Vector Config
# ############################
# VLEN: The number of bits in a single vector register
VLEN = WIDTH
# ELEN: The number of bits in an element within a vector register
ELEN = 8

# ############################
# Tensor Config
# ############################

# ############################
instructions = [
    (ALLOC, 0, 4, 0), # 100 00000000 0000
    (COPY_TO_MEM, 0, 0, 0),
    (COPY_TO_MEM, 1, 1, 0),
    (COPY_FROM_MEM, 0, 1, 0),
    (COPY_FROM_MEM, 1, 0, 0),
]
encoded_instructions = [
    encode(inst[1], inst[2], inst[3], inst[0]) for inst in instructions
]

INST_NUM = len(encoded_instructions)

@df.region()
def top():

    mem_inst: Stream[uint8, 4]
    mem_rs1: Stream[uint8, 4]
    mem_rs2: Stream[uint8, 4]

    # send vector operands
    # vec_inst: Stream[uint8, 4]
    # vs1: Stream[uint256, 4]
    # vs2: Stream[uint256, 4]
    # vd: Stream[uint256, 4]

    @df.kernel(mapping=[1])
    def DECODER(insts: uint32[INST_NUM]):
        for i in range(INST_NUM):
            inst: uint32 = insts[i]
            rs1_addr: UInt(8) = inst[4:12]
            rs2_addr: UInt(8) = inst[12:20]
            rd_addr: UInt(8) = inst[20:28]
            opcode: UInt(4) = inst[28:32]
            if opcode == ALLOC or opcode == COPY_TO_MEM or opcode == COPY_FROM_MEM:
                mem_inst_: UInt(8) = opcode
                mem_inst.put(mem_inst_)
                mem_rs1.put(rs1_addr)
                mem_rs2.put(rs2_addr)
            elif opcode == FLUSH:
                mem_inst_: UInt(8) = opcode
                mem_inst.put(mem_inst_)

            # elif opcode == VEC_ADD:

    @df.kernel(mapping=[1])
    def MMU(input_mem: uint256[HEIGHT], output_mem: uint256[HEIGHT]):
        memory: uint256[HEIGHT]
        bit_map: UInt(2)[HEIGHT] = 0
        for _ in range(INST_NUM):
            mem_inst_: UInt(8) = mem_inst.get()
            if mem_inst_ == ALLOC:
                starting_addr: UInt(8) = mem_rs1.get()
                size: UInt(8) = mem_rs2.get()
                # lack validity check
                for i in range(size):
                    bit_map[starting_addr + i] = CLEAN
            elif mem_inst_ == COPY_TO_MEM:
                src_addr: UInt(8) = mem_rs1.get()
                dst_addr: UInt(8) = mem_rs2.get()
                memory[dst_addr] = input_mem[src_addr]
            elif mem_inst_ == COPY_FROM_MEM:
                src_addr: UInt(8) = mem_rs1.get()
                dst_addr: UInt(8) = mem_rs2.get()
                output_mem[dst_addr] = memory[src_addr]

    # @df.kernel(mapping=[1])
    # def VEC():
    #     for _ in range(INST_NUM):
    #         operand1: uint256 = vs1.get()
    #         operand2: uint256 = vs2.get()

    #         add_8b: uint256
    #         for i in allo.grid(VLEN // 8, name="arith_8"):
    #             scalar1: int8 = operand1[i * 8 : (i + 1) * 8]
    #             scalar2: int8 = operand2[i * 8 : (i + 1) * 8]
    #             add_8b[i * 8 : (i + 1) * 8] = scalar2 + scalar1

    #         vd.put(add_8b)


input_ = np.zeros((HEIGHT, VLEN // 8)).astype(np.int8)
output_ = np.zeros((HEIGHT, VLEN // 8)).astype(np.int8)
input_[0, :] = np.random.randint(0, 8, (VLEN // 8,)).astype(np.int8)
input_[1, :] = np.random.randint(0, 8, (VLEN // 8,)).astype(np.int8)
print(input_)

packed_input = np.ascontiguousarray(input_).view(np_256).reshape((HEIGHT,))
packed_output = np.ascontiguousarray(output_).view(np_256).reshape((HEIGHT,))

mod = df.build(top, target="simulator")
print("start simulation")
mod(np.array(encoded_instructions, dtype=np.uint32), packed_input, packed_output)
unpacked_output = packed_output.view(np.int8).reshape(HEIGHT, VLEN // 8)
print(unpacked_output)