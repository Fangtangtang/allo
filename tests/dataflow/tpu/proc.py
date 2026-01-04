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
    return value


# ---- opcode
NOP = 0
ALLOC = 1  # starting line: addr1; line cnt: imm
FLUSH = 2  # polling untill all WAIT_VEC/WAIT_TENSOR returned and set to CLEAN
COPY_TO_MEM = 3  # copy by line
COPY_FROM_MEM = 4

VEC_ADD = 8

# -- MMU
GET_VEC_OPERAND = 5
SET_VEC_RESULT = 6

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

WAIT_COMPUTE = 1  # wait computation results
WAIT_MEM = 2  # wait MMU to consume

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
    (ALLOC, 0, 4, 0),  # 100 00000000 0000
    (COPY_TO_MEM, 0, 0, 0),
    (COPY_TO_MEM, 1, 0, 2),
    (COPY_FROM_MEM, 0, 0, 1),
    (COPY_FROM_MEM, 2, 0, 0),
]

# instructions = [
#     (ALLOC, 0, 4, 0),  # 100 00000000 0000
#     (COPY_TO_MEM, 0, 0, 0),
#     (COPY_TO_MEM, 1, 0, 1),
#     (VEC_ADD, 0, 1, 2),
#     (COPY_FROM_MEM, 2, 0, 0),
# ]

# instructions = [
#     (ALLOC, 0, 4, 0),  # 100 00000000 0000
#     (COPY_TO_MEM, 0, 0, 0),
#     (COPY_TO_MEM, 1, 0, 1),
#     (VEC_ADD, 0, 1, 2),
#     (VEC_ADD, 1, 2, 3),
#     (COPY_FROM_MEM, 2, 0, 0),
#     (COPY_FROM_MEM, 3, 0, 1),
# ]

encoded_instructions = [
    encode(inst[1], inst[2], inst[3], inst[0]) for inst in instructions
]

INST_NUM = len(encoded_instructions)


@df.region()
def top(
    Inst: uint32[INST_NUM], INPUT_MEM: uint256[HEIGHT], OUTPUT_MEM: uint256[HEIGHT]
):

    mem_inst: Stream[uint8, 4]
    mem_rs1: Stream[uint8, 4]
    mem_rs2: Stream[uint8, 4]

    # send vector operands
    vec_inst: Stream[uint8, 4]
    vs1: Stream[uint256, 4]
    vs2: Stream[uint256, 4]
    vd: Stream[uint256, 4]

    @df.kernel(mapping=[1], args=[Inst])
    def DECODER(insts: uint32[INST_NUM]):
        nop: UInt(8) = NOP
        for i in range(INST_NUM):
            inst: uint32 = insts[i]
            rs1_addr: UInt(8) = inst[20:28]
            rs2_addr: UInt(8) = inst[12:20]
            rd_addr: UInt(8) = inst[4:12]
            opcode: UInt(4) = inst[0:4]
            if opcode == ALLOC:
                mem_inst_: UInt(8) = opcode
                mem_inst.put(mem_inst_)
                mem_rs1.put(rs1_addr)
                mem_rs2.put(rs2_addr)
                vec_inst.put(nop)
            elif opcode == COPY_TO_MEM or opcode == COPY_FROM_MEM:
                mem_inst_: UInt(8) = opcode
                mem_inst.put(mem_inst_)
                mem_rs1.put(rs1_addr)
                mem_rs2.put(rd_addr)
                vec_inst.put(nop)
            elif opcode == FLUSH:
                mem_inst_: UInt(8) = opcode
                mem_inst.put(mem_inst_)
                vec_inst.put(nop)
            elif opcode == VEC_ADD:
                # get operand
                mem_inst_: UInt(8) = GET_VEC_OPERAND
                mem_inst.put(mem_inst_)
                mem_rs1.put(rs1_addr)
                mem_rs2.put(rs2_addr)

                # compute op
                vec_inst_: UInt(8) = VADD
                vec_inst.put(vec_inst_)

                # write back
                mem_rs2.put(rd_addr)

    @df.kernel(mapping=[1], args=[INPUT_MEM, OUTPUT_MEM])
    def MMU(input_mem: uint256[HEIGHT], output_mem: uint256[HEIGHT]):
        memory: uint256[HEIGHT]
        bit_map: UInt(4)[HEIGHT] = 0  # [state code (2 bits)][ring idx (2 bits)]

        vec_queue: uint256[4]  # reorder buffer
        vec_queue_bit_map: UInt(2)[4] = 0
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
            elif mem_inst_ == GET_VEC_OPERAND:
                vs1_addr: UInt(8) = mem_rs1.get()
                vs2_addr: UInt(8) = mem_rs2.get()
                # send two operand to vector processor
                if bit_map[vs1_addr] == CLEAN:
                    vs1.put(memory[vs1_addr])
                else:
                    # TODO
                    pass
                if bit_map[vs2_addr] == CLEAN:
                    vs2.put(memory[vs2_addr])
                else:
                    # TODO
                    pass
                # TODO: mark vd as WAIT_VEC and move forward
                vd_addr: UInt(8) = mem_rs2.get()
                vec_result: uint256 = vd.get()
                bit_map[vd_addr] = CLEAN
                memory[vd_addr] = vec_result

    @df.kernel(mapping=[1])
    def VEC():
        for _ in range(INST_NUM):
            vec_inst_: UInt(8) = vec_inst.get()
            if vec_inst_ == VADD:
                operand1: uint256 = vs1.get()
                operand2: uint256 = vs2.get()

                add_8b: uint256
                for i in allo.grid(VLEN // 8, name="arith_8"):
                    scalar1: int8 = operand1[i * 8 : (i + 1) * 8]
                    scalar2: int8 = operand2[i * 8 : (i + 1) * 8]
                    add_8b[i * 8 : (i + 1) * 8] = scalar2 + scalar1

                vd.put(add_8b)

    # TODO: tensor processor from `tests/dataflow/test_multi_cache_gemm.py`
    # @df.kernel(mapping=[1])
    # def TENSOR():
    #     pass


input_ = np.zeros((HEIGHT, VLEN // 8)).astype(np.int8)
output_ = np.zeros((HEIGHT, VLEN // 8)).astype(np.int8)
input_[0, :] = np.random.randint(0, 8, (VLEN // 8,)).astype(np.int8)
input_[1, :] = np.random.randint(0, 8, (VLEN // 8,)).astype(np.int8)
print(input_)

packed_input = np.ascontiguousarray(input_).view(np_256).reshape((HEIGHT,))
packed_output = np.ascontiguousarray(output_).view(np_256).reshape((HEIGHT,))
packed_instructions = np.array(encoded_instructions, dtype=np.uint32)

mod = df.build(top, target="simulator")
print("start simulation")

mod(packed_instructions, packed_input, packed_output)
unpacked_output = packed_output.view(np.int8).reshape(HEIGHT, VLEN // 8)
print(unpacked_output)

if hls.is_available("vitis_hls"):
    s = df.customize(top)
    print("Starting Test...")
    mod = s.build(
        target="vitis_hls",
        mode="hw_emu",
        project=f"proc.prj",
        wrap_io=False,
    )
    output_ = np.zeros((HEIGHT, VLEN // 8)).astype(np.int8)
    packed_output = np.ascontiguousarray(output_).view(np_256).reshape((HEIGHT,))
    mod(packed_instructions, packed_input, packed_output)
    unpacked_output = packed_output.view(np.int8).reshape(HEIGHT, VLEN // 8)
    print(unpacked_output)
