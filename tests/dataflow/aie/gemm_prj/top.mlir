module {
  aie.device(npu1_4col) {
    func.func private @fill_zeros_bf16_64_64_vector(memref<64x64xbf16>)
    func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
    func.func private @add_bf16_vector(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
    func.func private @matmul_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    aie.objectfifo @fifo_0(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_0_3, %tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_2(%mem_tile_1_1 dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_1_2, %tile_1_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_3(%shim_noc_tile_2_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_4(%mem_tile_2_1 dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_2_2, %tile_2_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_5(%shim_noc_tile_3_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_6(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_1_4, %tile_0_4}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_7(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_8(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_0_2, %tile_2_2, %tile_1_2, %tile_0_4}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_9(%shim_noc_tile_2_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_10(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_1_4, %tile_0_3, %tile_1_3, %tile_2_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_11(%shim_noc_tile_3_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_12(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_13(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_14(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x2x64x64xbf16>> 
    aie.objectfifo @fifo_15(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_16(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_17(%mem_tile_1_1 dimensionsToStream [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x2x64x64xbf16>> 
    aie.objectfifo @fifo_18(%tile_2_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_19(%tile_2_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_20(%mem_tile_2_1 dimensionsToStream [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1x2x64x64xbf16>> 
    aie.objectfifo @fifo_21(%tile_0_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_22(%tile_1_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_23(%mem_tile_1_1 dimensionsToStream [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1x2x64x64xbf16>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_7] -> [@fifo_6]([] [])
    aie.objectfifo.link [@fifo_9] -> [@fifo_8]([] [])
    aie.objectfifo.link [@fifo_12, @fifo_13] -> [@fifo_14]([0, 4096] [])
    aie.objectfifo.link [@fifo_3] -> [@fifo_2]([] [])
    aie.objectfifo.link [@fifo_11] -> [@fifo_10]([] [])
    aie.objectfifo.link [@fifo_15, @fifo_16] -> [@fifo_17]([0, 4096] [])
    aie.objectfifo.link [@fifo_21, @fifo_22] -> [@fifo_23]([0, 4096] [])
    aie.objectfifo.link [@fifo_5] -> [@fifo_4]([] [])
    aie.objectfifo.link [@fifo_18, @fifo_19] -> [@fifo_20]([0, 4096] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_12(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_12(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_13(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_13(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_15(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_15(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_16(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_16(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_18(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_18(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_19(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_19(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_21(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_21(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_22(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_22(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_3, Trace : 0>
      aie.packet_dest<%shim_noc_tile_1_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(2) {
      aie.packet_source<%mem_tile_0_1, Trace : 0>
      aie.packet_dest<%shim_noc_tile_1_0, DMA : 1>
    } {keep_pkt_header = true}
    aiex.runtime_sequence(%arg0: memref<131072xbf16>, %arg1: memref<65536xbf16>, %arg2: memref<131072xbf16>) {
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 3 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 3 : i32, value = 1 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 3 : i32, value = 559107915 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 3 : i32, value = 622466850 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 3 : i32, value = 74273 : ui32}
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 3 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 3 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 65536 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 1 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673700 : ui32, arg_idx = 3 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 1 : i32, row = 0 : i32, value = 15 : ui32}
      aiex.npu.write32 {address = 606416 : ui32, column = 0 : i32, row = 1 : i32, value = 2627534848 : ui32}
      aiex.npu.write32 {address = 606420 : ui32, column = 0 : i32, row = 1 : i32, value = 12290 : ui32}
      aiex.npu.write32 {address = 606432 : ui32, column = 0 : i32, row = 1 : i32, value = 1549292624 : ui32}
      aiex.npu.write32 {address = 606436 : ui32, column = 0 : i32, row = 1 : i32, value = 1818780768 : ui32}
      aiex.npu.write32 {address = 724736 : ui32, column = 0 : i32, row = 1 : i32, value = 16780832 : ui32}
      aiex.npu.write32 {address = 724740 : ui32, column = 0 : i32, row = 1 : i32, value = 572588802 : ui32}
      aiex.npu.write32 {address = 606208 : ui32, column = 0 : i32, row = 1 : i32, value = 7424 : ui32}
      aiex.npu.writebd {bd_id = 14 : i32, buffer_length = 65536 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 2 : i32, packet_type = 3 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673668 : ui32, arg_idx = 3 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 1 : i32, row = 0 : i32, value = 14 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 1 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 1 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 1 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 8, 64, 64][32768, 64, 512, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<131072xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 8, 64, 64][32768, 64, 512, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_1} : memref<131072xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[1, 0, 0, 0][1, 8, 64, 64][32768, 64, 512, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<131072xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[1, 0, 0, 0][1, 8, 64, 64][32768, 64, 512, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_3} : memref<131072xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[2, 0, 0, 0][1, 8, 64, 64][32768, 64, 512, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<131072xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[2, 0, 0, 0][1, 8, 64, 64][32768, 64, 512, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<131072xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[3, 0, 0, 0][1, 8, 64, 64][32768, 64, 512, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_7} : memref<131072xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[3, 0, 0, 0][1, 8, 64, 64][32768, 64, 512, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_7} : memref<131072xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 8, 64, 64][64, 16384, 256, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_9} : memref<131072xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 0, 0, 0][1, 8, 64, 64][64, 16384, 256, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_9} : memref<131072xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 0, 0, 0][1, 8, 64, 64][64, 16384, 256, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<131072xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 0, 0, 0][1, 8, 64, 64][64, 16384, 256, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_11} : memref<131072xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 2, 64, 64][16384, 64, 256, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_14} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][1, 2, 64, 64][16384, 64, 256, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_14} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[1, 0, 0, 0][1, 2, 64, 64][16384, 64, 256, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_17} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[1, 2, 0, 0][1, 2, 64, 64][16384, 64, 256, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_17} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[2, 0, 0, 0][1, 2, 64, 64][16384, 64, 256, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_20} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[2, 2, 0, 0][1, 2, 64, 64][16384, 64, 256, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_20} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[3, 0, 0, 0][1, 2, 64, 64][16384, 64, 256, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_23} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[3, 2, 0, 0][1, 2, 64, 64][16384, 64, 256, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_23} : memref<65536xbf16>
      aiex.npu.dma_wait {symbol = @fifo_14}
      aiex.npu.dma_wait {symbol = @fifo_17}
      aiex.npu.dma_wait {symbol = @fifo_20}
      aiex.npu.dma_wait {symbol = @fifo_23}
      aiex.npu.dma_wait {symbol = @fifo_14}
      aiex.npu.dma_wait {symbol = @fifo_17}
      aiex.npu.dma_wait {symbol = @fifo_20}
      aiex.npu.dma_wait {symbol = @fifo_23}
      aie.end
    }
  }
}
