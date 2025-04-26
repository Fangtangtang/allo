# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie import call_mlir

LyA = Layout("S0R")
LyB = Layout("RS1")
LyC = Layout("S0S1")

Ty = int16
M, N, K = 16, 16, 16

def _test_gemm_1D():    
    P0 = 1

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N], C: Ty[M, N] @ LyA):
            # `C + allo.matmul(A, B)`` leads to confusing `i17` intermediate result
            C[:, :] =  allo.add(C, allo.matmul(A, B))

    mod = df.build(top, target="aie")
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")

def test_gemm_mlir(projrct_dir:str):
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    call_mlir(
       projrct_dir, Ty, 4096, A, B, C
    )
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")

if __name__ == "__main__":
    _test_gemm_1D()
    test_gemm_mlir("top.prj")

'''
An valid mlir implementation

---
module {
  aie.device(npu1_4col) {
    func.func private @matmul_scalar_i16_i16(memref<16x16xi16>, memref<16x16xi16>, memref<16x16xi16>)
    %tile_shim0 = aie.tile(0, 0)
    %tile_shim1 = aie.tile(1, 0)
    %tile_shim2 = aie.tile(2, 0)
    %tile_mem0 = aie.tile(0, 1)
    %tile_mem1 = aie.tile(1, 1)
    %tile_mem2 = aie.tile(2, 1)
    %tile_comp_gemm_0 = aie.tile(0, 2)
    %tile_comp_gemm_0_buf0 = aie.buffer(%tile_comp_gemm_0) : memref<16x16xi16>
    %tile_comp_gemm_0_buf1 = aie.buffer(%tile_comp_gemm_0) : memref<16x16xi16>
    aie.objectfifo @in_shim_A(%tile_shim0, {%tile_mem0}, 2 : i32) : !aie.objectfifo<memref<1x1x16x16xi16>>
    aie.objectfifo @in_mem_A_0R(%tile_mem0, {%tile_comp_gemm_0}, 2 : i32) : !aie.objectfifo<memref<16x16xi16>>
    aie.objectfifo.link [@in_shim_A] -> [@in_mem_A_0R]([] [0])
    aie.objectfifo @in_shim_B(%tile_shim1, {%tile_mem1}, 2 : i32) : !aie.objectfifo<memref<1x1x16x16xi16>>
    aie.objectfifo @in_mem_B_RR(%tile_mem1, {%tile_comp_gemm_0}, 2 : i32) : !aie.objectfifo<memref<16x16xi16>>
    aie.objectfifo.link [@in_shim_B] -> [@in_mem_B_RR]([] [0])
    aie.objectfifo @out_shim_C(%tile_mem2, {%tile_shim2}, 2 : i32) : !aie.objectfifo<memref<1x1x16x16xi16>>
    aie.objectfifo @out_mem_C_0R(%tile_comp_gemm_0, {%tile_mem2}, 2 : i32) : !aie.objectfifo<memref<16x16xi16>>
    aie.objectfifo.link [@out_mem_C_0R] -> [@out_shim_C]([0] [])
    %core_0_2_tile_comp_gemm_0 = aie.core(%tile_comp_gemm_0) {
      %global_c0 = arith.constant 0 : index
      %global_c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %global_c0 to %c9223372036854775807 step %global_c1 {
        %fifo0 = aie.objectfifo.acquire @in_mem_A_0R(Consume, 1) : !aie.objectfifosubview<memref<16x16xi16>>
        %local0 = aie.objectfifo.subview.access %fifo0[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
        %fifo1 = aie.objectfifo.acquire @in_mem_B_RR(Consume, 1) : !aie.objectfifosubview<memref<16x16xi16>>
        %local1 = aie.objectfifo.subview.access %fifo1[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
        %fifo_out0 = aie.objectfifo.acquire @out_mem_C_0R(Produce, 1) : !aie.objectfifosubview<memref<16x16xi16>>
        %local_out0 = aie.objectfifo.subview.access %fifo_out0[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
        %c0_i16 = arith.constant 0 : i16
        %c0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1 = arith.constant 1 : index
        scf.for %arg3 = %c0 to %c16 step %c1 {
          %c0_4 = arith.constant 0 : index
          %c16_5 = arith.constant 16 : index
          %c1_6 = arith.constant 1 : index
          scf.for %arg4 = %c0_4 to %c16_5 step %c1_6 {
            memref.store %c0_i16, %tile_comp_gemm_0_buf0[%arg3, %arg4] : memref<16x16xi16>
          }
        }
        func.call @matmul_scalar_i16_i16(%local0, %local1, %tile_comp_gemm_0_buf0) : (memref<16x16xi16>, memref<16x16xi16>, memref<16x16xi16>) -> ()
        aie.objectfifo.release @in_mem_A_0R(Consume, 1)
        %fifo0_C = aie.objectfifo.acquire @in_mem_A_0R(Consume, 1) : !aie.objectfifosubview<memref<16x16xi16>>
        %local0_C = aie.objectfifo.subview.access %fifo0_C[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
        %c0_1 = arith.constant 0 : index
        %c16_2 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg3 = %c0_1 to %c16_2 step %c1_3 {
          %c0_4 = arith.constant 0 : index
          %c16_5 = arith.constant 16 : index
          %c1_6 = arith.constant 1 : index
          scf.for %arg4 = %c0_4 to %c16_5 step %c1_6 {
            %0 = memref.load %local0_C[%arg3, %arg4] : memref<16x16xi16>
            %1 = memref.load %tile_comp_gemm_0_buf0[%arg3, %arg4] : memref<16x16xi16>
            %2 = arith.addi %0, %1 : i16
            memref.store %2, %tile_comp_gemm_0_buf1[%arg3, %arg4] : memref<16x16xi16>
          }
        }
        memref.copy %tile_comp_gemm_0_buf1, %local_out0 {to = "C"} : memref<16x16xi16> to memref<16x16xi16>
        aie.objectfifo.release @in_mem_A_0R(Consume, 1)
        aie.objectfifo.release @in_mem_B_RR(Consume, 1)
        aie.objectfifo.release @out_mem_C_0R(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}
    aiex.runtime_sequence(%arg0: memref<16x16xi16>, %arg1: memref<16x16xi16>, %arg2: memref<16x16xi16>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 256, 16, 1]) {id = 0 : i64, issue_token = true, metadata = @in_shim_A} : memref<16x16xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 16, 16][0, 0, 16, 1]) {id = 1 : i64, issue_token = true, metadata = @in_shim_B} : memref<16x16xi16>
      // reuse port to send C
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 16, 16][0, 256, 16, 1]) {id = 0 : i64, issue_token = true, metadata = @in_shim_A} : memref<16x16xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 16, 16][0, 256, 16, 1]) {id = 2 : i64, metadata = @out_shim_C} : memref<16x16xi16>
      aiex.npu.dma_wait {symbol = @in_shim_A}
      aiex.npu.dma_wait {symbol = @in_shim_B}
      aiex.npu.dma_wait {symbol = @out_shim_C}
    }
  }
}
'''