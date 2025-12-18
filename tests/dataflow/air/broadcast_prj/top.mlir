module {
  func.func @top(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1_0) args(%arg6=%arg0, %arg7=%arg1) : memref<1024xi32>, memref<1024xi32> {
      %c0 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %alloc = memref.alloc() : memref<1024xi32, 2 : i32>
      %alloc_2 = memref.alloc() : memref<1024xi32, 2 : i32>
      %c1024 = arith.constant 1024 : index
      air.dma_memcpy_nd (%alloc[] [] [], %arg6[%c0] [%c1024] [%c1_1]) : (memref<1024xi32, 2 : i32>, memref<1024xi32>)
      %c1_i32 = arith.constant 1 : i32
      %alloc_3 = memref.alloc() : memref<i32, 2 : i32>
      linalg.fill ins(%c1_i32 : i32) outs(%alloc_3 : memref<i32, 2 : i32>)
      %alloc_4 = memref.alloc() : memref<1024xi32, 2 : i32>
      linalg.broadcast ins(%alloc_3 : memref<i32, 2 : i32>) outs(%alloc_4 : memref<1024xi32, 2 : i32>) dimensions = [0] 
      linalg.add ins(%alloc, %alloc_4 : memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32>) outs(%alloc_2 : memref<1024xi32, 2 : i32>)
      %c1024_5 = arith.constant 1024 : index
      air.dma_memcpy_nd (%arg7[%c0] [%c1024_5] [%c1_1], %alloc_2[] [] []) : (memref<1024xi32>, memref<1024xi32, 2 : i32>)
      memref.dealloc %alloc : memref<1024xi32, 2 : i32>
      memref.dealloc %alloc_2 : memref<1024xi32, 2 : i32>
      memref.dealloc %alloc_3 : memref<i32, 2 : i32>
      memref.dealloc %alloc_4 : memref<1024xi32, 2 : i32>
    }
    return
  }
}
