module {
  func.func @core_0(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {df.kernel, itypes = "ss", otypes = "", stypes = "__", tag = "core_()"} {
    %c1_i32 = arith.constant 1 : i32
    %alloc = memref.alloc() : memref<i32>
    linalg.fill ins(%c1_i32 : i32) outs(%alloc : memref<i32>)
    %alloc_0 = memref.alloc() : memref<1024xi32>
    linalg.broadcast ins(%alloc : memref<i32>) outs(%alloc_0 : memref<1024xi32>) dimensions = [0] 
    %alloc_1 = memref.alloc() : memref<1024xi32>
    linalg.add {op_name = "add_0"} ins(%arg0, %alloc_0 : memref<1024xi32>, memref<1024xi32>) outs(%alloc_1 : memref<1024xi32>)
    memref.copy %alloc_1, %arg1 {to = "B"} : memref<1024xi32> to memref<1024xi32>
    return
  }
  func.func @top(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {dataflow, itypes = "ss"} {
    return
  }
}
