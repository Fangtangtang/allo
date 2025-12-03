module {
  aie.device(npu1_4col) {
    memref.global "public" @fifo_5 : memref<1x1x1x16xi32>
    memref.global "public" @fifo_3 : memref<1x1x1x16xi32>
    memref.global "public" @fifo_1 : memref<1x1x1x16xi32>
    func.func private @add_i32_vector(memref<16xi32>, memref<16xi32>, memref<16xi32>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %fifo_4_cons_buff_0 = aie.buffer(%mem_tile_2_1) : memref<16xi32> 
    %fifo_4_cons_buff_1 = aie.buffer(%mem_tile_2_1) : memref<16xi32> 
    %fifo_4_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "fifo_4_cons_prod_lock_0"}
    %fifo_4_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "fifo_4_cons_cons_lock_0"}
    %fifo_4_buff_0 = aie.buffer(%tile_0_2) : memref<16xi32> 
    %fifo_4_buff_1 = aie.buffer(%tile_0_2) : memref<16xi32> 
    %fifo_4_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "fifo_4_prod_lock_0"}
    %fifo_4_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "fifo_4_cons_lock_0"}
    %fifo_2_cons_buff_0 = aie.buffer(%mem_tile_1_1) : memref<16xi32> 
    %fifo_2_cons_buff_1 = aie.buffer(%mem_tile_1_1) : memref<16xi32> 
    %fifo_2_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "fifo_2_cons_prod_lock_0"}
    %fifo_2_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "fifo_2_cons_cons_lock_0"}
    %fifo_2_buff_0 = aie.buffer(%tile_0_3) : memref<16xi32> 
    %fifo_2_buff_1 = aie.buffer(%tile_0_3) : memref<16xi32> 
    %fifo_2_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "fifo_2_prod_lock_0"}
    %fifo_2_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "fifo_2_cons_lock_0"}
    %fifo_1_cons_buff_0 = aie.buffer(%mem_tile_0_1) : memref<1x1x1x16xi32> 
    %fifo_1_cons_buff_1 = aie.buffer(%mem_tile_0_1) : memref<1x1x1x16xi32> 
    %fifo_1_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "fifo_1_cons_prod_lock_0"}
    %fifo_1_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "fifo_1_cons_cons_lock_0"}
    %fifo_0_cons_buff_0 = aie.buffer(%tile_0_3) : memref<16xi32> 
    %fifo_0_cons_buff_1 = aie.buffer(%tile_0_3) : memref<16xi32> 
    %fifo_0_cons_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "fifo_0_cons_prod_lock_0"}
    %fifo_0_cons_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "fifo_0_cons_cons_lock_0"}
    %pipe_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "pipe_cons_lock_0"}
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    %_anonymous0 = aie.buffer(%tile_0_3) : memref<i32> 
    %_anonymous1 = aie.buffer(%tile_0_3) : memref<16xi32> 
    %_anonymous3 = aie.buffer(%tile_0_2) : memref<i32> 
    %_anonymous4 = aie.buffer(%tile_0_2) : memref<16xi32> 
    %core_0_3 = aie.core(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      memref.store %c1_i32, %_anonymous0[] : memref<i32>
      %c0_0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1_1 = arith.constant 1 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %4 = memref.load %_anonymous0[] : memref<i32>
      memref.store %4, %_anonymous1[%2] : memref<16xi32>
      %5 = arith.addi %2, %c1_1 : index
      cf.br ^bb3(%5 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @add_i32_vector(%fifo_0_cons_buff_0, %_anonymous1, %fifo_2_buff_0) {lib = "add_i32_vector"} : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
      aie.use_lock(%pipe_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_lock_0, Release, 1)
      memref.store %c1_i32, %_anonymous0[] : memref<i32>
      %c0_2 = arith.constant 0 : index
      %c16_3 = arith.constant 16 : index
      %c1_4 = arith.constant 1 : index
      cf.br ^bb6(%c0_2 : index)
    ^bb6(%6: index):  // 2 preds: ^bb5, ^bb7
      %7 = arith.cmpi slt, %6, %c16_3 : index
      cf.cond_br %7, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      %8 = memref.load %_anonymous0[] : memref<i32>
      memref.store %8, %_anonymous1[%6] : memref<16xi32>
      %9 = arith.addi %6, %c1_4 : index
      cf.br ^bb6(%9 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @add_i32_vector(%fifo_0_cons_buff_1, %_anonymous1, %fifo_2_buff_1) {lib = "add_i32_vector"} : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
      aie.use_lock(%pipe_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_lock_0, Release, 1)
      %10 = arith.addi %0, %c2 : index
      cf.br ^bb1(%10 : index)
    ^bb9:  // pred: ^bb1
      memref.store %c1_i32, %_anonymous0[] : memref<i32>
      %c0_5 = arith.constant 0 : index
      %c16_6 = arith.constant 16 : index
      %c1_7 = arith.constant 1 : index
      cf.br ^bb10(%c0_5 : index)
    ^bb10(%11: index):  // 2 preds: ^bb9, ^bb11
      %12 = arith.cmpi slt, %11, %c16_6 : index
      cf.cond_br %12, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      %13 = memref.load %_anonymous0[] : memref<i32>
      memref.store %13, %_anonymous1[%11] : memref<16xi32>
      %14 = arith.addi %11, %c1_7 : index
      cf.br ^bb10(%14 : index)
    ^bb12:  // pred: ^bb10
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @add_i32_vector(%fifo_0_cons_buff_0, %_anonymous1, %fifo_2_buff_0) {lib = "add_i32_vector"} : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
      aie.use_lock(%pipe_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "external0.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      aie.use_lock(%pipe_cons_lock_0, AcquireGreaterEqual, 1)
      memref.store %c1_i32, %_anonymous3[] : memref<i32>
      %c0_0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1_1 = arith.constant 1 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %4 = memref.load %_anonymous3[] : memref<i32>
      memref.store %4, %_anonymous4[%2] : memref<16xi32>
      %5 = arith.addi %2, %c1_1 : index
      cf.br ^bb3(%5 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%fifo_4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @add_i32_vector(%fifo_0_cons_buff_0, %_anonymous4, %fifo_4_buff_0) {lib = "add_i32_vector"} : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_lock_0, Release, 1)
      aie.use_lock(%pipe_cons_lock_0, AcquireGreaterEqual, 1)
      memref.store %c1_i32, %_anonymous3[] : memref<i32>
      %c0_2 = arith.constant 0 : index
      %c16_3 = arith.constant 16 : index
      %c1_4 = arith.constant 1 : index
      cf.br ^bb6(%c0_2 : index)
    ^bb6(%6: index):  // 2 preds: ^bb5, ^bb7
      %7 = arith.cmpi slt, %6, %c16_3 : index
      cf.cond_br %7, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      %8 = memref.load %_anonymous3[] : memref<i32>
      memref.store %8, %_anonymous4[%6] : memref<16xi32>
      %9 = arith.addi %6, %c1_4 : index
      cf.br ^bb6(%9 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%fifo_4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @add_i32_vector(%fifo_0_cons_buff_1, %_anonymous4, %fifo_4_buff_1) {lib = "add_i32_vector"} : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_lock_0, Release, 1)
      %10 = arith.addi %0, %c2 : index
      cf.br ^bb1(%10 : index)
    ^bb9:  // pred: ^bb1
      aie.use_lock(%pipe_cons_lock_0, AcquireGreaterEqual, 1)
      memref.store %c1_i32, %_anonymous3[] : memref<i32>
      %c0_5 = arith.constant 0 : index
      %c16_6 = arith.constant 16 : index
      %c1_7 = arith.constant 1 : index
      cf.br ^bb10(%c0_5 : index)
    ^bb10(%11: index):  // 2 preds: ^bb9, ^bb11
      %12 = arith.cmpi slt, %11, %c16_6 : index
      cf.cond_br %12, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      %13 = memref.load %_anonymous3[] : memref<i32>
      memref.store %13, %_anonymous4[%11] : memref<16xi32>
      %14 = arith.addi %11, %c1_7 : index
      cf.br ^bb10(%14 : index)
    ^bb12:  // pred: ^bb10
      aie.use_lock(%fifo_4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @add_i32_vector(%fifo_0_cons_buff_0, %_anonymous4, %fifo_4_buff_0) {lib = "add_i32_vector"} : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "external0.o"}
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_0 : memref<1x1x1x16xi32>, 0, 16) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_1 : memref<1x1x1x16xi32>, 0, 16) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_0 : memref<1x1x1x16xi32>, 0, 16) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_1 : memref<1x1x1x16xi32>, 0, 16) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_cons_buff_0 : memref<16xi32>, 0, 16) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_cons_buff_1 : memref<16xi32>, 0, 16) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_buff_0 : memref<16xi32>, 0, 16) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_buff_1 : memref<16xi32>, 0, 16) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @fifo_1(MM2S, 0, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_0 : memref<16xi32>, 0, 16) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_1 : memref<16xi32>, 0, 16) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_0 : memref<16xi32>, 0, 16) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_1 : memref<16xi32>, 0, 16) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @fifo_3(S2MM, 0, 1)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_buff_0 : memref<16xi32>, 0, 16) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_buff_1 : memref<16xi32>, 0, 16) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_0 : memref<16xi32>, 0, 16) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_1 : memref<16xi32>, 0, 16) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_0 : memref<16xi32>, 0, 16) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_1 : memref<16xi32>, 0, 16) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @fifo_5(S2MM, 0, 2)
    aiex.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @fifo_3}
      aiex.npu.dma_wait {symbol = @fifo_5}
      aie.end
    }
  }
}
