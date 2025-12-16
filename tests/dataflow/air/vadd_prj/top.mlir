#map = affine_map<()[s0, s1] -> (s0 + s1 * 1024)>
module {
  func.func @vector_add(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.herd @herd_0  tile (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1_0) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<1024xf32>, memref<1024xf32>, memref<1024xf32> {
      %alloc = memref.alloc() : memref<1024xf32, 2 : i32>
      %alloc_1 = memref.alloc() : memref<1024xf32, 2 : i32>
      %alloc_2 = memref.alloc() : memref<1024xf32, 2 : i32>
      %c0 = arith.constant 0 : index
      %c1024_3 = arith.constant 1024 : index
      %0 = affine.apply #map()[%c0, %arg4]
      %c1024_4 = arith.constant 1024 : index
      %c1_5 = arith.constant 1 : index
      air.dma_memcpy_nd (%alloc[] [] [], %arg7[%0] [%c1024_4] [%c1_5]) : (memref<1024xf32, 2 : i32>, memref<1024xf32>)
      %c1024_6 = arith.constant 1024 : index
      %c1_7 = arith.constant 1 : index
      air.dma_memcpy_nd (%alloc_1[] [] [], %arg8[%0] [%c1024_6] [%c1_7]) : (memref<1024xf32, 2 : i32>, memref<1024xf32>)
      linalg.add {op_name = "add_0"} ins(%alloc, %alloc_1 : memref<1024xf32, 2 : i32>, memref<1024xf32, 2 : i32>) outs(%alloc_2 : memref<1024xf32, 2 : i32>)
      %c1024_11 = arith.constant 1024 : index
      %c1_12 = arith.constant 1 : index
      air.dma_memcpy_nd (%arg9[%0] [%c1024_11] [%c1_12], %alloc_2[] [] []) : (memref<1024xf32>, memref<1024xf32, 2 : i32>)
      memref.dealloc %alloc : memref<1024xf32, 2 : i32>
      memref.dealloc %alloc_1 : memref<1024xf32, 2 : i32>
      memref.dealloc %alloc_2 : memref<1024xf32, 2 : i32>
    }
    return
  }
}

