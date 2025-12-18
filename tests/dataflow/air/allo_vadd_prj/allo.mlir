module {
  func.func @core_0(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    %alloc = memref.alloc() : memref<1024xf32>
    linalg.add {op_name = "add_0"} ins(%arg0, %arg1 : memref<1024xf32>, memref<1024xf32>) outs(%alloc : memref<1024xf32>)
    memref.copy %alloc, %arg2 {to = "C"} : memref<1024xf32> to memref<1024xf32>
    return
  }
  func.func @top(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) attributes {dataflow, itypes = "___"} {
    return
  }
}