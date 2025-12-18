module {
  func.func @core_0(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {df.kernel, itypes = "ss", otypes = "", stypes = "__", tag = "core_()"} {
    memref.copy %arg0, %arg1 {to = "B"} : memref<1024xi32> to memref<1024xi32>
    return
  }
  func.func @top(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {dataflow, itypes = "ss"} {
    return
  }
}
