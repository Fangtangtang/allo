module {
  aie.device(npu1_4col) {
    func.func private @add_i32_vector(memref<16xi32>, memref<16xi32>, memref<16xi32>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @pipe(%tile_0_3, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>> 
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<16xi32>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x1x16xi32>> 
    aie.objectfifo @fifo_2(%tile_0_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16xi32>> 
    aie.objectfifo @fifo_3(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x1x1x16xi32>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_2] -> [@fifo_3]([] [])
    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<16xi32> 
    %buffer_0_2_0 = aie.buffer(%tile_0_2) : memref<i32> 
    %buffer_0_2_1 = aie.buffer(%tile_0_2) : memref<16xi32> 
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %2 = aie.objectfifo.acquire @pipe(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        memref.copy %1, %3 : memref<16xi32> to memref<16xi32>
        aie.objectfifo.release @pipe(Produce, 1)
        aie.objectfifo.release @fifo_0(Consume, 1)
      }
      aie.end
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        affine.store %c1_i32, %buffer_0_2_0[] : memref<i32>
        affine.for %arg1 = 0 to 16 {
          %4 = affine.load %buffer_0_2_0[] : memref<i32>
          affine.store %4, %buffer_0_2_1[%arg1] : memref<16xi32>
        }
        %2 = aie.objectfifo.acquire @fifo_2(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @add_i32_vector(%1, %buffer_0_2_1, %3) {lib = "add_i32_vector"} : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
        aie.objectfifo.release @pipe(Consume, 1)
        aie.objectfifo.release @fifo_2(Produce, 1)
      }
      aie.end
    } {link_with = "external1.o"}
    aiex.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @fifo_3}
      aie.end
    }
  }
}
