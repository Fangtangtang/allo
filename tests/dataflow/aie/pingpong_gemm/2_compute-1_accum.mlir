module {
  aie.device(npu1_4col) {
    func.func private @matmul_scalar_i16_i16(memref<16x16xi16>, memref<16x16xi16>, memref<16x16xi16>)
    %tile_shim0 = aie.tile(0, 0)
    %tile_shim1 = aie.tile(1, 0)
    %tile_mem0 = aie.tile(0, 1)
    %tile_mem1 = aie.tile(1, 1)
    %tile_comp_gemm_0_0_0 = aie.tile(0, 2)
    %tile_comp_gemm_1_0_0 = aie.tile(1, 2)
    %tile_comp_gemm_acc = aie.tile(0, 3)

    %tile_comp_gemm_1_0_0_buf0 = aie.buffer(%tile_comp_gemm_1_0_0) : memref<16x16xi16>
    
    aie.objectfifo @in_shim_A(%tile_shim0, {%tile_mem0}, 2 : i32) : !aie.objectfifo<memref<16x32xi16>>
    aie.objectfifo @in_mem_A_00(%tile_mem0, {%tile_comp_gemm_0_0_0}, 2 : i32) : !aie.objectfifo<memref<16x16xi16>>
    aie.objectfifo @in_mem_A_01(%tile_mem0, {%tile_comp_gemm_1_0_0}, 2 : i32) : !aie.objectfifo<memref<16x16xi16>>
    aie.objectfifo.link [@in_shim_A] -> [@in_mem_A_00, @in_mem_A_01]([] [0, 256])
    aie.objectfifo @in_shim_B(%tile_shim1, {%tile_mem1}, 2 : i32) : !aie.objectfifo<memref<32x16xi16>>
    aie.objectfifo @in_mem_B_00(%tile_mem1, {%tile_comp_gemm_0_0_0}, 2 : i32) : !aie.objectfifo<memref<16x16xi16>>
    aie.objectfifo @in_mem_B_10(%tile_mem1, {%tile_comp_gemm_1_0_0}, 2 : i32) : !aie.objectfifo<memref<16x16xi16>>
    aie.objectfifo.link [@in_shim_B] -> [@in_mem_B_00, @in_mem_B_10]([] [0, 256])

    aie.objectfifo @out_shim_C(%tile_mem0, {%tile_shim0}, 2 : i32) : !aie.objectfifo<memref<16x16xi16>>
    aie.objectfifo @out_mem_C_00(%tile_comp_gemm_acc, {%tile_mem0}, 2 : i32) : !aie.objectfifo<memref<16x16xi16>>
    aie.objectfifo.link [@out_mem_C_00] -> [@out_shim_C]([0] [])

    aie.objectfifo @pipe_0_0_0(%tile_comp_gemm_0_0_0, {%tile_comp_gemm_acc}, 2 : i32) : !aie.objectfifo<memref<16x16xi16>>
    aie.objectfifo @pipe_1_0_0(%tile_comp_gemm_1_0_0, {%tile_comp_gemm_acc}, 2 : i32) : !aie.objectfifo<memref<16x16xi16>>
   
    %core_0_2_tile_comp_gemm_0_0_0 = aie.core(%tile_comp_gemm_0_0_0) {
      %global_c0 = arith.constant 0 : index
      %global_c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c0_i16 = arith.constant 0 : i16
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %global_c0 to %c9223372036854775807 step %global_c1 {
        %fifo0 = aie.objectfifo.acquire @in_mem_A_00(Consume, 1) : !aie.objectfifosubview<memref<16x16xi16>>
        %local0 = aie.objectfifo.subview.access %fifo0[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
        %fifo1 = aie.objectfifo.acquire @in_mem_B_00(Consume, 1) : !aie.objectfifosubview<memref<16x16xi16>>
        %local1 = aie.objectfifo.subview.access %fifo1[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
        %fifo_pipe_0_0_0 = aie.objectfifo.acquire @pipe_0_0_0(Produce, 1) : !aie.objectfifosubview<memref<16x16xi16>>
        %local_pipe_0_0_0 = aie.objectfifo.subview.access %fifo_pipe_0_0_0[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
        scf.for %arg4 = %global_c0 to %c16 step %global_c1 {
          scf.for %arg5 = %global_c0 to %c16 step %global_c1 {
            memref.store %c0_i16, %local_pipe_0_0_0[%arg4, %arg5] : memref<16x16xi16>
          }
        }
        func.call @matmul_scalar_i16_i16(%local0, %local1, %local_pipe_0_0_0) : (memref<16x16xi16>, memref<16x16xi16>, memref<16x16xi16>) -> ()
        aie.objectfifo.release @pipe_0_0_0(Produce, 1)
        aie.objectfifo.release @in_mem_A_00(Consume, 1)
        aie.objectfifo.release @in_mem_B_00(Consume, 1)
      }
      aie.end
    } {link_with = "external.o"}

    %core_0_3_tile_comp_gemm_1_0_0 = aie.core(%tile_comp_gemm_1_0_0) {
      %global_c0 = arith.constant 0 : index
      %global_c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c0_i16 = arith.constant 0 : i16
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %global_c0 to %c9223372036854775807 step %global_c1 {
        %fifo0 = aie.objectfifo.acquire @in_mem_A_01(Consume, 1) : !aie.objectfifosubview<memref<16x16xi16>>
        %local0 = aie.objectfifo.subview.access %fifo0[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
        %fifo1 = aie.objectfifo.acquire @in_mem_B_10(Consume, 1) : !aie.objectfifosubview<memref<16x16xi16>>
        %local1 = aie.objectfifo.subview.access %fifo1[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
        %fifo_pipe_1_0_0 = aie.objectfifo.acquire @pipe_1_0_0(Produce, 1) : !aie.objectfifosubview<memref<16x16xi16>>
        %local_pipe_1_0_0 = aie.objectfifo.subview.access %fifo_pipe_1_0_0[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
        scf.for %arg4 = %global_c0 to %c16 step %global_c1 {
          scf.for %arg5 = %global_c0 to %c16 step %global_c1 {
            memref.store %c0_i16, %local_pipe_1_0_0[%arg4, %arg5] : memref<16x16xi16>
          }
        }
        func.call @matmul_scalar_i16_i16(%local0, %local1, %local_pipe_1_0_0) : (memref<16x16xi16>, memref<16x16xi16>, memref<16x16xi16>) -> ()
        aie.objectfifo.release @pipe_1_0_0(Produce, 1)
        aie.objectfifo.release @in_mem_A_01(Consume, 1)
        aie.objectfifo.release @in_mem_B_10(Consume, 1)        
      }
      aie.end
    } {link_with = "external.o"}

    %core_0_4_tile_comp_gemm_acc = aie.core(%tile_comp_gemm_acc) {
      %global_c0 = arith.constant 0 : index
      %global_c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %global_c0 to %c9223372036854775807 step %global_c1 {
        %fifo_out0 = aie.objectfifo.acquire @out_mem_C_00(Produce, 1) : !aie.objectfifosubview<memref<16x16xi16>>
        %local_out0 = aie.objectfifo.subview.access %fifo_out0[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
        %fifo_pipe_0_0_0 = aie.objectfifo.acquire @pipe_0_0_0(Consume, 1) : !aie.objectfifosubview<memref<16x16xi16>>
        %local_pipe_0_0_0 = aie.objectfifo.subview.access %fifo_pipe_0_0_0[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
        aie.objectfifo.release @pipe_0_0_0(Consume, 1)
        %fifo_pipe_1_0_0 = aie.objectfifo.acquire @pipe_1_0_0(Consume, 1) : !aie.objectfifosubview<memref<16x16xi16>>
        %local_pipe_1_0_0 = aie.objectfifo.subview.access %fifo_pipe_1_0_0[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
        aie.objectfifo.release @pipe_1_0_0(Consume, 1)
        scf.for %arg4 = %global_c0 to %c16 step %global_c1 {
          scf.for %arg5 = %global_c0 to %c16 step %global_c1 {
            %1 = memref.load %local_pipe_0_0_0[%arg4, %arg5] : memref<16x16xi16>
            %2 = memref.load %local_pipe_1_0_0[%arg4, %arg5] : memref<16x16xi16>
            %3 = arith.addi %1, %2 : i16
            memref.store %3, %local_out0[%arg4, %arg5] : memref<16x16xi16>
          }
        }
        aie.objectfifo.release @out_mem_C_00(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}

    aiex.runtime_sequence(%arg0: memref<16x32xi16>, %arg1: memref<32x16xi16>, %arg2: memref<16x16xi16>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 2, 16, 16][512, 16, 32, 1]) {id = 0 : i64, issue_token = true, metadata = @in_shim_A} : memref<16x32xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][2, 1, 16, 16][256, 16, 16, 1]) {id = 1 : i64, issue_token = true, metadata = @in_shim_B} : memref<32x16xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 16, 16][256, 16, 16, 1]) {id = 2 : i64, metadata = @out_shim_C} : memref<16x16xi16>
      aiex.npu.dma_wait {symbol = @in_shim_A}
      aiex.npu.dma_wait {symbol = @in_shim_B}
      aiex.npu.dma_wait {symbol = @out_shim_C}
    }
  }
}