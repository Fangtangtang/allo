# modified from: https://github.com/Xilinx/mlir-aie/blob/main/programming_examples/basic/matrix_multiplication/single_core/single_core.py
import argparse
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
import aie.utils.trace as trace_utils
from aie.helpers.dialects.ext.scf import _for as range_

dtype_map = {
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("-M", type=int, default=1024)
    argparser.add_argument("-K", type=int, default=1024)
    argparser.add_argument("-N", type=int, default=1024)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=64)
    argparser.add_argument("--dtype_in", type=str, choices=["i8", "i16"], default="i16")
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["i8", "i16", "f32", "i32"],
        default="i16",
    )
    argparser.add_argument("--trace_size", type=int, default=65536)
    args = argparser.parse_args()
    my_matmul(
        args.M,
        args.K,
        args.N,
        args.m,
        args.k,
        args.n,
        args.dtype_in,
        args.dtype_out,
        args.trace_size,
    )


def my_matmul(M, K, N, m, k, n, dtype_in_str, dtype_out_str, trace_size):

    enable_tracing = True if trace_size > 0 else False

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    A_sz = M * K
    B_sz = K * N
    C_sz = M * N

    with mlir_mod_ctx() as ctx:

        C_sz_in_bytes = C_sz * np.dtype(dtype_out).itemsize

        @device(AIEDevice.npu1_4col)
        def device_body():
            # Tile declarations
            shim_tile_0 = tile(1, 0)
            shim_tile = tile(2, 0)
            mem_tile = tile(0, 1)
            mem_tile_3 = tile(3, 1)
            compute_tile2_col, compute_tile2_row = 0, 2
            compute_tile2 = tile(compute_tile2_col, compute_tile2_row)
            compute_tile3 = tile(0, 3)
            compute_tile4 = tile(0, 4)
            compute_tile5 = tile(0, 5)

            # fixme: use the tiles you want to trace
            tiles_to_trace = [
                compute_tile2,
                compute_tile3,
                compute_tile4,
                compute_tile5,
                mem_tile,
                # mem_tile_3,
                # shim_tile_0,
            ]
            if trace_size > 0:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile)

            # To/from AIE-array data movement
            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[dtype_in]],
                np.ndarray[(B_sz,), np.dtype[dtype_in]],
                np.ndarray[(C_sz,), np.dtype[dtype_out]],
            )
            def sequence(A, B, C):
                if enable_tracing:
                    trace_utils.configure_packet_tracing_aie2(
                        tiles_to_trace,
                        shim_tile,
                        trace_size,
                        C_sz_in_bytes,
                        coretile_events=[
                            # captures input A (PORT_RUNNING_0, at port number 1, master for inputs)
                            trace_utils.PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_0,
                                port_number=1,
                                master=True,
                            ),
                            # captures input B (PORT_RUNNING_1, at port number 2, master for inputs)
                            trace_utils.PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_1,
                                port_number=2,
                                master=True,
                            ),
                            # captures output C (PORT_RUNNING_2, at port number 1, slave for outputs)
                            trace_utils.PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_2,
                                port_number=1,
                                master=False,
                            ),
                            trace_utils.CoreEvent.INSTR_EVENT_0,
                            trace_utils.CoreEvent.INSTR_EVENT_1,
                            trace_utils.CoreEvent.MEMORY_STALL,
                            trace_utils.CoreEvent.LOCK_STALL,
                            trace_utils.CoreEvent.INSTR_VECTOR,
                        ],
                        memtile_events=[
                            trace_utils.MemTilePortEvent(
                                trace_utils.MemTileEvent.PORT_RUNNING_0, 0, True
                            ),  # master(0)
                            trace_utils.MemTilePortEvent(
                                trace_utils.MemTileEvent.PORT_RUNNING_1, 14, False
                            ),  # slave(14/ north1)
                            trace_utils.MemTilePortEvent(
                                trace_utils.MemTileEvent.PORT_RUNNING_2, 0, False
                            ),  # slave(0)
                            trace_utils.MemTilePortEvent(
                                trace_utils.MemTileEvent.PORT_RUNNING_3, 1, False
                            ),  # slave(1)
                            trace_utils.MemTilePortEvent(
                                trace_utils.MemTileEvent.PORT_RUNNING_4, 2, False
                            ),  # slave(2)
                            trace_utils.MemTilePortEvent(
                                trace_utils.MemTileEvent.PORT_RUNNING_5, 3, False
                            ),  # slave(3)
                            trace_utils.MemTilePortEvent(
                                trace_utils.MemTileEvent.PORT_RUNNING_6, 1, True
                            ),  # slave(4)
                            trace_utils.MemTilePortEvent(
                                trace_utils.MemTileEvent.PORT_RUNNING_7, 2, True
                            ),  # slave(1)
                        ],
                    )

    print(ctx.module)


if __name__ == "__main__":
    main()
