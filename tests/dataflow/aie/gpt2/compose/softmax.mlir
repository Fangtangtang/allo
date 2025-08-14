module {
  aie.device(npu1_4col) {
    func.func private @softmax_bfloat16(memref<8x512xbf16>, memref<8x512xbf16>)
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
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_3 = aie.tile(3, 3)
    %tile_3_4 = aie.tile(3, 4)
    %tile_3_5 = aie.tile(3, 5)
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_1(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_2(%mem_tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_3(%mem_tile_0_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_4(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x4x8x512xbf16>> 
    aie.objectfifo @fifo_5(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_6(%mem_tile_1_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_7(%mem_tile_1_1, {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_8(%mem_tile_1_1, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_9(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x4x8x512xbf16>> 
    aie.objectfifo @fifo_10(%mem_tile_2_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_11(%mem_tile_2_1, {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_12(%mem_tile_2_1, {%tile_2_4}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_13(%mem_tile_2_1, {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_14(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x4x8x512xbf16>> 
    aie.objectfifo @fifo_15(%mem_tile_3_1, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_16(%mem_tile_3_1, {%tile_3_3}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_17(%mem_tile_3_1, {%tile_3_4}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_18(%mem_tile_3_1, {%tile_3_5}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_19(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1x4x8x512xbf16>> 
    aie.objectfifo @fifo_20(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_21(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_22(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_23(%tile_0_5, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_24(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x4x8x512xbf16>> 
    aie.objectfifo @fifo_25(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_26(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_27(%tile_1_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_28(%tile_1_5, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_29(%mem_tile_1_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x4x8x512xbf16>> 
    aie.objectfifo @fifo_30(%tile_2_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_31(%tile_2_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_32(%tile_2_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_33(%tile_2_5, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_34(%mem_tile_2_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x4x8x512xbf16>> 
    aie.objectfifo @fifo_35(%tile_3_2, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_36(%tile_3_3, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_37(%tile_3_4, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_38(%tile_3_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<8x512xbf16>> 
    aie.objectfifo @fifo_39(%mem_tile_3_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x4x8x512xbf16>> 
    aie.objectfifo.link [@fifo_4] -> [@fifo_0, @fifo_1, @fifo_2, @fifo_3]([] [0, 4096, 8192, 12288])
    aie.objectfifo.link [@fifo_20, @fifo_21, @fifo_22, @fifo_23] -> [@fifo_24]([0, 4096, 8192, 12288] [])
    aie.objectfifo.link [@fifo_9] -> [@fifo_5, @fifo_6, @fifo_7, @fifo_8]([] [0, 4096, 8192, 12288])
    aie.objectfifo.link [@fifo_25, @fifo_26, @fifo_27, @fifo_28] -> [@fifo_29]([0, 4096, 8192, 12288] [])
    aie.objectfifo.link [@fifo_14] -> [@fifo_10, @fifo_11, @fifo_12, @fifo_13]([] [0, 4096, 8192, 12288])
    aie.objectfifo.link [@fifo_30, @fifo_31, @fifo_32, @fifo_33] -> [@fifo_34]([0, 4096, 8192, 12288] [])
    aie.objectfifo.link [@fifo_19] -> [@fifo_15, @fifo_16, @fifo_17, @fifo_18]([] [0, 4096, 8192, 12288])
    aie.objectfifo.link [@fifo_35, @fifo_36, @fifo_37, @fifo_38] -> [@fifo_39]([0, 4096, 8192, 12288] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_20(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_20(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_21(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_21(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_22(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_22(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_23(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_23(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_5(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_25(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_5(Consume, 1)
        aie.objectfifo.release @fifo_25(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_26(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_26(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_7(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_27(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_7(Consume, 1)
        aie.objectfifo.release @fifo_27(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_28(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_28(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_30(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_30(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_11(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_31(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_11(Consume, 1)
        aie.objectfifo.release @fifo_31(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_32(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_12(Consume, 1)
        aie.objectfifo.release @fifo_32(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_13(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_33(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_13(Consume, 1)
        aie.objectfifo.release @fifo_33(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_15(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_35(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_15(Consume, 1)
        aie.objectfifo.release @fifo_35(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_16(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_36(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_16(Consume, 1)
        aie.objectfifo.release @fifo_36(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_17(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_37(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_17(Consume, 1)
        aie.objectfifo.release @fifo_37(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_18(Consume, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        %2 = aie.objectfifo.acquire @fifo_38(Produce, 1) : !aie.objectfifosubview<memref<8x512xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x512xbf16>> -> memref<8x512xbf16>
        func.call @softmax_bfloat16(%1, %3) : (memref<8x512xbf16>, memref<8x512xbf16>) -> ()
        aie.objectfifo.release @fifo_18(Consume, 1)
        aie.objectfifo.release @fifo_38(Produce, 1)
      }
      aie.end
    } {link_with = "s_external0.o"}
    aiex.runtime_sequence(%arg0: memref<262144xbf16>, %arg1: memref<262144xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_4} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_4} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 32, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_4} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 48, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_4} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_9} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_9} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 36, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_9} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 52, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_9} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_14} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_14} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 40, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_14} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 56, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_14} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_19} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_19} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 44, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_19} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 60, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_19} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_24} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_24} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 32, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_24} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 48, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_24} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_29} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_29} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 36, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_29} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 52, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_29} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_34} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_34} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 40, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_34} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 56, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_34} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_39} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_39} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 44, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_39} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 60, 0, 0][1, 4, 8, 512][0, 4096, 512, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_39} : memref<262144xbf16>
      aiex.npu.dma_wait {symbol = @fifo_24}
      aiex.npu.dma_wait {symbol = @fifo_29}
      aiex.npu.dma_wait {symbol = @fifo_34}
      aiex.npu.dma_wait {symbol = @fifo_39}
      aiex.npu.dma_wait {symbol = @fifo_24}
      aiex.npu.dma_wait {symbol = @fifo_29}
      aiex.npu.dma_wait {symbol = @fifo_34}
      aiex.npu.dma_wait {symbol = @fifo_39}
      aiex.npu.dma_wait {symbol = @fifo_24}
      aiex.npu.dma_wait {symbol = @fifo_29}
      aiex.npu.dma_wait {symbol = @fifo_34}
      aiex.npu.dma_wait {symbol = @fifo_39}
      aiex.npu.dma_wait {symbol = @fifo_24}
      aiex.npu.dma_wait {symbol = @fifo_29}
      aiex.npu.dma_wait {symbol = @fifo_34}
      aiex.npu.dma_wait {symbol = @fifo_39}
      aie.end
    }
  }
}
