module {
  aie.device(npu1_4col) {
    func.func private @fill_zeros_bf16_32_32_vector(memref<32x32xbf16>)
    func.func private @matmul_bf16_bf16(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @pipe_A_0(%tile_0_3, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @pipe_B_0(%tile_0_3, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_2(%mem_tile_1_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_3(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_4(%tile_0_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_5(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_6(%tile_0_2, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_7(%mem_tile_3_1, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_3] -> [@fifo_2]([] [])
    aie.objectfifo.link [@fifo_4] -> [@fifo_5]([] [])
    aie.objectfifo.link [@fifo_6] -> [@fifo_7]([] [])
    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<32x32xbf16> 
    %buffer_0_2_0 = aie.buffer(%tile_0_2) : memref<32x32xbf16> 
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_4(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) {lib = "matmul_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        %6 = aie.objectfifo.acquire @pipe_A_0(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        memref.copy %3, %7 : memref<32x32xbf16> to memref<32x32xbf16>
        aie.objectfifo.release @pipe_A_0(Produce, 1)
        %8 = aie.objectfifo.acquire @pipe_B_0(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        memref.copy %5, %9 : memref<32x32xbf16> to memref<32x32xbf16>
        aie.objectfifo.release @pipe_B_0(Produce, 1)
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_4(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_A_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %2 = aie.objectfifo.acquire @pipe_B_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_6(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        func.call @matmul_bf16_bf16(%1, %3, %5) {lib = "matmul_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @pipe_A_0(Consume, 1)
        aie.objectfifo.release @pipe_B_0(Consume, 1)
        aie.objectfifo.release @fifo_6(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<1024xbf16>, %arg1: memref<4096xbf16>, %arg2: memref<1024xbf16>, %arg3: memref<4096xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][4, 1, 32, 32][0, 0, 32, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][4, 1, 32, 32][0, 0, 32, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][4, 1, 32, 32][32, 32, 128, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<4096xbf16>
      aiex.npu.dma_memcpy_nd(%arg3[0, 0, 0, 0][4, 1, 32, 32][32, 32, 128, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_7} : memref<4096xbf16>
      aiex.npu.dma_wait {symbol = @fifo_5}
      aiex.npu.dma_wait {symbol = @fifo_7}
      aie.end
    }
  }
}
