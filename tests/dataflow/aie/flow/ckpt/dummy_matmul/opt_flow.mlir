module {
  aie.device(npu1_4col) {
    memref.global "public" @fifo_7 : memref<1x1x32x32xbf16>
    memref.global "public" @fifo_5 : memref<1x1x32x32xbf16>
    memref.global "public" @fifo_3 : memref<1x1x32x32xbf16>
    memref.global "public" @fifo_1 : memref<1x1x32x32xbf16>
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
    %fifo_6_cons_buff_0 = aie.buffer(%mem_tile_3_1) : memref<32x32xbf16> 
    %fifo_6_cons_buff_1 = aie.buffer(%mem_tile_3_1) : memref<32x32xbf16> 
    %fifo_6_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "fifo_6_cons_prod_lock_0"}
    %fifo_6_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "fifo_6_cons_cons_lock_0"}
    %fifo_6_buff_0 = aie.buffer(%tile_0_2) : memref<32x32xbf16> 
    %fifo_6_buff_1 = aie.buffer(%tile_0_2) : memref<32x32xbf16> 
    %fifo_6_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "fifo_6_prod_lock_0"}
    %fifo_6_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "fifo_6_cons_lock_0"}
    %fifo_4_cons_buff_0 = aie.buffer(%mem_tile_2_1) : memref<32x32xbf16> 
    %fifo_4_cons_buff_1 = aie.buffer(%mem_tile_2_1) : memref<32x32xbf16> 
    %fifo_4_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "fifo_4_cons_prod_lock_0"}
    %fifo_4_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "fifo_4_cons_cons_lock_0"}
    %fifo_4_buff_0 = aie.buffer(%tile_0_3) : memref<32x32xbf16> 
    %fifo_4_buff_1 = aie.buffer(%tile_0_3) : memref<32x32xbf16> 
    %fifo_4_prod_lock_0 = aie.lock(%tile_0_3, 8) {init = 2 : i32, sym_name = "fifo_4_prod_lock_0"}
    %fifo_4_cons_lock_0 = aie.lock(%tile_0_3, 9) {init = 0 : i32, sym_name = "fifo_4_cons_lock_0"}
    %fifo_3_cons_buff_0 = aie.buffer(%mem_tile_1_1) : memref<1x1x32x32xbf16> 
    %fifo_3_cons_buff_1 = aie.buffer(%mem_tile_1_1) : memref<1x1x32x32xbf16> 
    %fifo_3_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "fifo_3_cons_prod_lock_0"}
    %fifo_3_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "fifo_3_cons_cons_lock_0"}
    %fifo_2_cons_buff_0 = aie.buffer(%tile_0_3) : memref<32x32xbf16> 
    %fifo_2_cons_buff_1 = aie.buffer(%tile_0_3) : memref<32x32xbf16> 
    %fifo_2_cons_prod_lock_0 = aie.lock(%tile_0_3, 6) {init = 2 : i32, sym_name = "fifo_2_cons_prod_lock_0"}
    %fifo_2_cons_cons_lock_0 = aie.lock(%tile_0_3, 7) {init = 0 : i32, sym_name = "fifo_2_cons_cons_lock_0"}
    %fifo_1_cons_buff_0 = aie.buffer(%mem_tile_0_1) : memref<1x1x32x32xbf16> 
    %fifo_1_cons_buff_1 = aie.buffer(%mem_tile_0_1) : memref<1x1x32x32xbf16> 
    %fifo_1_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "fifo_1_cons_prod_lock_0"}
    %fifo_1_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "fifo_1_cons_cons_lock_0"}
    %fifo_0_cons_buff_0 = aie.buffer(%tile_0_3) : memref<32x32xbf16> 
    %fifo_0_cons_buff_1 = aie.buffer(%tile_0_3) : memref<32x32xbf16> 
    %fifo_0_cons_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "fifo_0_cons_prod_lock_0"}
    %fifo_0_cons_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "fifo_0_cons_cons_lock_0"}
    %pipe_B_0_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "pipe_B_0_cons_lock_0"}
    %pipe_A_0_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "pipe_A_0_cons_lock_0"}
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_0_3, DMA : 1)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_4_buff_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%pipe_A_0_cons_lock_0, Release, 1)
      aie.use_lock(%pipe_B_0_cons_lock_0, Release, 1)
      func.call @matmul_bf16_bf16(%fifo_0_cons_buff_0, %fifo_2_cons_buff_0, %fifo_4_buff_0) {lib = "matmul_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_4_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_4_buff_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%pipe_A_0_cons_lock_0, Release, 1)
      aie.use_lock(%pipe_B_0_cons_lock_0, Release, 1)
      func.call @matmul_bf16_bf16(%fifo_0_cons_buff_1, %fifo_2_cons_buff_1, %fifo_4_buff_1) {lib = "matmul_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_4_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_4_buff_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%pipe_A_0_cons_lock_0, Release, 1)
      aie.use_lock(%pipe_B_0_cons_lock_0, Release, 1)
      func.call @matmul_bf16_bf16(%fifo_0_cons_buff_0, %fifo_2_cons_buff_0, %fifo_4_buff_0) {lib = "matmul_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_4_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "external0.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%pipe_A_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%pipe_B_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_6_buff_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      func.call @matmul_bf16_bf16(%fifo_0_cons_buff_0, %fifo_2_cons_buff_0, %fifo_6_buff_0) {lib = "matmul_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_6_cons_lock_0, Release, 1)
      aie.use_lock(%pipe_A_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%pipe_B_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_6_buff_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      func.call @matmul_bf16_bf16(%fifo_0_cons_buff_1, %fifo_2_cons_buff_1, %fifo_6_buff_1) {lib = "matmul_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_6_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%pipe_A_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%pipe_B_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_6_buff_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      func.call @matmul_bf16_bf16(%fifo_0_cons_buff_0, %fifo_2_cons_buff_0, %fifo_6_buff_0) {lib = "matmul_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_6_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "external0.o"}
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_0 : memref<1x1x32x32xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_1 : memref<1x1x32x32xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_0 : memref<1x1x32x32xbf16>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_1 : memref<1x1x32x32xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_cons_buff_0 : memref<32x32xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_cons_buff_1 : memref<32x32xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_0 : memref<32x32xbf16>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_1 : memref<32x32xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_buff_0 : memref<32x32xbf16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_buff_1 : memref<32x32xbf16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @fifo_1(MM2S, 0, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_3_cons_buff_0 : memref<1x1x32x32xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_3_cons_buff_1 : memref<1x1x32x32xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_3_cons_buff_0 : memref<1x1x32x32xbf16>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_3_cons_buff_1 : memref<1x1x32x32xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @fifo_3(MM2S, 0, 1)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_0 : memref<32x32xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_1 : memref<32x32xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_0 : memref<32x32xbf16>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_1 : memref<32x32xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @fifo_5(S2MM, 0, 2)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_buff_0 : memref<32x32xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_buff_1 : memref<32x32xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<32x32xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<32x32xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<32x32xbf16>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<32x32xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @fifo_7(S2MM, 0, 3)
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
