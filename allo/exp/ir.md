<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->
# SPMW IR

The SPMW IR is built on top of MLIR. 
Each MLIR `module` compiles from a top-level funtion decorated with`spmw.unit`. Worker kernels become separate `func.func` entries. Sharding information is encoded in `allo.grid_map` ops with `sharding` and `grid` attributes. Streams become `allo.stream_global` globals accessed via `allo.put_stream_global` / `allo.get_stream_global`. Worker IDs are obtained through generated `xxx.mesh.get_wid` private functions.

## SPMW MLIR Operations

## Examples
Please refer to [syntax document](./syntax.md) for the programming interface.

### Sharding

#### 1D Sharding

Both tensors sharded along the worker axis:

```python
@spmw.unit()
def top(A: int32[1024], B: int32[1024]):
    @spmw.work(grid=[4])
    def core():
        x = spmw.axes()
        local_A = A.shard([x])
        local_B: int32[256] = B.shard([x])
        local_B[:] = local_A + 1
```

```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func @top(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {itypes = "ss", otypes = ""} {
    allo.grid_map(%arg0, %arg1) sharding = [[0], [0]] grid = [4] { // both tensors shard along the 0 dim of the grid
    ^bb0(%arg2: memref<256xi32>, %arg3: memref<256xi32>):
      func.call @top.core(%arg2, %arg3) : (memref<256xi32>, memref<256xi32>) -> ()
    } : memref<1024xi32>, memref<1024xi32>
    return
  }
  // the worker function definition
  func.func private @top.core.mesh.get_wid() -> index
  func.func @top.core(%arg0: memref<256xi32>, %arg1: memref<256xi32>) attributes {itypes = "ss", otypes = ""} {
    %0 = call @top.core.mesh.get_wid() : () -> index // get work id, always called at the entry of each work function
    %alloc = memref.alloc() {signed} : memref<256xi33>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : memref<256xi32>) outs(%alloc : memref<256xi33>) attrs =  {signed} {
    ^bb0(%in: i32, %out: i33):
      %1 = arith.extsi %in {signed} : i32 to i33
      linalg.yield %1 : i33
    }
    %c1_i33 = arith.constant 1 : i33
    %alloc_0 = memref.alloc() {signed} : memref<256xi33>
    linalg.fill ins(%c1_i33 : i33) outs(%alloc_0 : memref<256xi33>)
    %alloc_1 = memref.alloc() {signed} : memref<256xi33>
    linalg.add ins(%alloc, %alloc_0 : memref<256xi33>, memref<256xi33>) outs(%alloc_1 : memref<256xi33>)
    %alloc_2 = memref.alloc() {signed} : memref<256xi32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_1 : memref<256xi33>) outs(%alloc_2 : memref<256xi32>) attrs =  {signed} {
    ^bb0(%in: i33, %out: i32):
      %1 = arith.trunci %in {signed} : i33 to i32
      linalg.yield %1 : i32
    }
    %subview = memref.subview %arg1[0] [256] [1] : memref<256xi32> to memref<256xi32, strided<[1]>>
    memref.copy %alloc_2, %subview : memref<256xi32> to memref<256xi32, strided<[1]>>
    return
  }
}
```

Input tensor replicated (`shard([None])`), output tensor sharded:

```python
@spmw.unit()
def top(A: int32[256], B: int32[1024]):
    @spmw.work(grid=[4])
    def core():
        x = spmw.axes()
        local_A = A.shard([None])
        local_B: int32[256] = B.shard([x])
        local_B[:] = local_A + 1
```

```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func @top(%arg0: memref<256xi32>, %arg1: memref<1024xi32>) attributes {itypes = "ss", otypes = ""} {
    allo.grid_map(%arg0, %arg1) sharding = [[-1], [0]] grid = [4] { // %arg0 replicated, %arg1 shard along the 0 dim of the grid
    ^bb0(%arg2: memref<256xi32>, %arg3: memref<256xi32>):
      func.call @top.core(%arg2, %arg3) : (memref<256xi32>, memref<256xi32>) -> ()
    } : memref<256xi32>, memref<1024xi32>
    return
  }
  func.func private @top.core.mesh.get_wid() -> index
  func.func @top.core(%arg0: memref<256xi32>, %arg1: memref<256xi32>) attributes {itypes = "ss", otypes = ""} {
    %0 = call @top.core.mesh.get_wid() : () -> index
    %alloc = memref.alloc() {signed} : memref<256xi33>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : memref<256xi32>) outs(%alloc : memref<256xi33>) attrs =  {signed} {
    ^bb0(%in: i32, %out: i33):
      %1 = arith.extsi %in {signed} : i32 to i33
      linalg.yield %1 : i33
    }
    %c1_i33 = arith.constant 1 : i33
    %alloc_0 = memref.alloc() {signed} : memref<256xi33>
    linalg.fill ins(%c1_i33 : i33) outs(%alloc_0 : memref<256xi33>)
    %alloc_1 = memref.alloc() {signed} : memref<256xi33>
    linalg.add ins(%alloc, %alloc_0 : memref<256xi33>, memref<256xi33>) outs(%alloc_1 : memref<256xi33>)
    %alloc_2 = memref.alloc() {signed} : memref<256xi32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_1 : memref<256xi33>) outs(%alloc_2 : memref<256xi32>) attrs =  {signed} {
    ^bb0(%in: i33, %out: i32):
      %1 = arith.trunci %in {signed} : i33 to i32
      linalg.yield %1 : i32
    }
    %subview = memref.subview %arg1[0] [256] [1] : memref<256xi32> to memref<256xi32, strided<[1]>>
    memref.copy %alloc_2, %subview : memref<256xi32> to memref<256xi32, strided<[1]>>
    return
  }
}
```

Input tensor passed directly (no explicit shard call), only output sharded:

```python
@spmw.unit()
def top(A: int32[256], B: int32[1024]):
    @spmw.work(grid=[4])
    def core():
        x = spmw.axes()
        local_B: int32[256] = B.shard([x])
        local_B[:] = A + 1
```

```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func @top(%arg0: memref<256xi32>, %arg1: memref<1024xi32>) attributes {itypes = "ss", otypes = ""} {
    allo.grid_map(%arg1) sharding = [[0]] grid = [4] { // %arg1 shard along the 0 dim of the grid
    ^bb0(%arg2: memref<256xi32>):
      func.call @top.core(%arg2, %arg0) : (memref<256xi32>, memref<256xi32>) -> ()
    } : memref<1024xi32>
    return
  }
  func.func private @top.core.mesh.get_wid() -> index
  func.func @top.core(%arg0: memref<256xi32>, %arg1: memref<256xi32>) attributes {itypes = "ss", otypes = ""} {
    %0 = call @top.core.mesh.get_wid() : () -> index
    %alloc = memref.alloc() {signed} : memref<256xi33>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg1 : memref<256xi32>) outs(%alloc : memref<256xi33>) attrs =  {signed} {
    ^bb0(%in: i32, %out: i33):
      %1 = arith.extsi %in {signed} : i32 to i33
      linalg.yield %1 : i33
    }
    %c1_i33 = arith.constant 1 : i33
    %alloc_0 = memref.alloc() {signed} : memref<256xi33>
    linalg.fill ins(%c1_i33 : i33) outs(%alloc_0 : memref<256xi33>)
    %alloc_1 = memref.alloc() {signed} : memref<256xi33>
    linalg.add ins(%alloc, %alloc_0 : memref<256xi33>, memref<256xi33>) outs(%alloc_1 : memref<256xi33>)
    %alloc_2 = memref.alloc() {signed} : memref<256xi32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_1 : memref<256xi33>) outs(%alloc_2 : memref<256xi32>) attrs =  {signed} {
    ^bb0(%in: i33, %out: i32):
      %1 = arith.trunci %in {signed} : i33 to i32
      linalg.yield %1 : i32
    }
    %subview = memref.subview %arg0[0] [256] [1] : memref<256xi32> to memref<256xi32, strided<[1]>>
    memref.copy %alloc_2, %subview : memref<256xi32> to memref<256xi32, strided<[1]>>
    return
  }
}
```

### 2D Sharding

2D tensors sharded along first axis only (`shard([x, None])`), 1D grid:

```python
M, N = 64, 64

@spmw.unit()
def top(A: int32[M, N], B: int32[M, N]):
    @spmw.work(grid=[4])
    def core():
        x = spmw.axes()
        local_A = A.shard([x, None])
        local_B = B.shard([x, None])
        local_B[:, :] = local_A + 1
```

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @top(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) attributes {itypes = "ss", otypes = ""} {
    allo.grid_map(%arg0, %arg1) sharding = [[0, -1], [0, -1]] grid = [4] { // The first dimension of the tensors shard along the 0 dim of the 1D grid
    ^bb0(%arg2: memref<16x64xi32>, %arg3: memref<16x64xi32>):
      func.call @top.core(%arg2, %arg3) : (memref<16x64xi32>, memref<16x64xi32>) -> ()
    } : memref<64x64xi32>, memref<64x64xi32>
    return
  }
  func.func private @top.core.mesh.get_wid() -> index
  func.func @top.core(%arg0: memref<16x64xi32>, %arg1: memref<16x64xi32>) attributes {itypes = "ss", otypes = ""} {
    %0 = call @top.core.mesh.get_wid() : () -> index
    %alloc = memref.alloc() {signed} : memref<16x64xi33>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<16x64xi32>) outs(%alloc : memref<16x64xi33>) attrs =  {signed} {
    ^bb0(%in: i32, %out: i33):
      %1 = arith.extsi %in {signed} : i32 to i33
      linalg.yield %1 : i33
    }
    %c1_i33 = arith.constant 1 : i33
    %alloc_0 = memref.alloc() {signed} : memref<16x64xi33>
    linalg.fill ins(%c1_i33 : i33) outs(%alloc_0 : memref<16x64xi33>)
    %alloc_1 = memref.alloc() {signed} : memref<16x64xi33>
    linalg.add ins(%alloc, %alloc_0 : memref<16x64xi33>, memref<16x64xi33>) outs(%alloc_1 : memref<16x64xi33>)
    %alloc_2 = memref.alloc() {signed} : memref<16x64xi32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_1 : memref<16x64xi33>) outs(%alloc_2 : memref<16x64xi32>) attrs =  {signed} {
    ^bb0(%in: i33, %out: i32):
      %1 = arith.trunci %in {signed} : i33 to i32
      linalg.yield %1 : i32
    }
    %subview = memref.subview %arg1[0, 0] [16, 64] [1, 1] : memref<16x64xi32> to memref<16x64xi32, strided<[64, 1]>>
    memref.copy %alloc_2, %subview : memref<16x64xi32> to memref<16x64xi32, strided<[64, 1]>>
    return
  }
}
```

2D tensors sharded along both axes (`shard([x, y])`), 2D grid — `get_wid` returns two indices:

```python
@spmw.unit()
def top(A: int32[64, 64], B: int32[64, 64]):
    @spmw.work(grid=[2, 2])
    def core():
        x, y = spmw.axes()
        local_A = A.shard([x, y])
        local_B = B.shard([x, y])
        local_B[:, :] = local_A + 1
```

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @top(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) attributes {itypes = "ss", otypes = ""} {
    allo.grid_map(%arg0, %arg1) sharding = [[0, 1], [0, 1]] grid = [2, 2] {
    ^bb0(%arg2: memref<32x32xi32>, %arg3: memref<32x32xi32>):
      func.call @top.core(%arg2, %arg3) : (memref<32x32xi32>, memref<32x32xi32>) -> ()
    } : memref<64x64xi32>, memref<64x64xi32>
    return
  }
  func.func private @top.core.mesh.get_wid() -> (index, index)
  func.func @top.core(%arg0: memref<32x32xi32>, %arg1: memref<32x32xi32>) attributes {itypes = "ss", otypes = ""} {
    %0:2 = call @top.core.mesh.get_wid() : () -> (index, index) // 2D grid, get work id returns two values
    %alloc = memref.alloc() {signed} : memref<32x32xi33>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<32x32xi32>) outs(%alloc : memref<32x32xi33>) attrs =  {signed} {
    ^bb0(%in: i32, %out: i33):
      %1 = arith.extsi %in {signed} : i32 to i33
      linalg.yield %1 : i33
    }
    %c1_i33 = arith.constant 1 : i33
    %alloc_0 = memref.alloc() {signed} : memref<32x32xi33>
    linalg.fill ins(%c1_i33 : i33) outs(%alloc_0 : memref<32x32xi33>)
    %alloc_1 = memref.alloc() {signed} : memref<32x32xi33>
    linalg.add ins(%alloc, %alloc_0 : memref<32x32xi33>, memref<32x32xi33>) outs(%alloc_1 : memref<32x32xi33>)
    %alloc_2 = memref.alloc() {signed} : memref<32x32xi32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_1 : memref<32x32xi33>) outs(%alloc_2 : memref<32x32xi32>) attrs =  {signed} {
    ^bb0(%in: i33, %out: i32):
      %1 = arith.trunci %in {signed} : i33 to i32
      linalg.yield %1 : i32
    }
    %subview = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
    memref.copy %alloc_2, %subview : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
    return
  }
}
```

### Get Worker ID

Worker ID accessed as `pi: ConstExpr[index] = x.id`; loop bounds expressed as affine maps over the worker ID:

```python
vlen = 1024
P = 4
tlen = vlen // P

@spmw.unit()
def top(A: int32[vlen], B: int32[vlen]):
    @spmw.work(grid=[P])
    def core():
        x = spmw.axes()
        pi: ConstExpr[index] = x.id
        for i in range(tlen * pi, tlen * (pi + 1)):
            B[i] = A[i] + 1
```

```mlir
#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> ((s0 + 1) * 256)>
module {
  func.func @top(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {itypes = "ss", otypes = ""} {
    allo.grid_map() sharding = [] grid = [4] {
      func.call @top.core(%arg0, %arg1) : (memref<1024xi32>, memref<1024xi32>) -> ()
    } : 
    return
  }
  func.func private @top.core.mesh.get_wid() -> index
  func.func @top.core(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {itypes = "ss", otypes = ""} {
    %0 = call @top.core.mesh.get_wid() : () -> index
    affine.for %arg2 = #map()[%0] to #map1()[%0] { // use the work id
      %1 = affine.load %arg0[%arg2] : memref<1024xi32>
      %2 = arith.extsi %1 {signed} : i32 to i33
      %c1_i33 = arith.constant 1 : i33
      %3 = arith.addi %2, %c1_i33 {signed} : i33
      %4 = arith.trunci %3 {signed} : i33 to i32
      affine.store %4, %arg1[%arg2] : memref<1024xi32>
    }
    return
  }
}
```


### Stream

A single scalar transferred between producer and consumer via a stream:

```python
@spmw.unit()
def top1(A: int32[16, 16], B: int32[16, 16]):
    pipe: Stream[int32]

    @spmw.work(grid=[1])
    def producer():
        pipe.put(A[0, 0])

    @spmw.work(grid=[1])
    def consumer():
        B[0, 0] = pipe.get()
```

```mlir
module {
  func.func @top1(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) attributes {itypes = "ss", otypes = ""} {
    allo.grid_map() sharding = [] grid = [1] {
      func.call @top1.producer(%arg0) : (memref<16x16xi32>) -> ()
    } : 
    allo.grid_map() sharding = [] grid = [1] {
      func.call @top1.consumer(%arg1) : (memref<16x16xi32>) -> ()
    } : 
    return
  }
  allo.stream_global @top1.pipe : !allo.stream<i32, 2> [] {_stream} // stream declared as a global symbol
  func.func private @top1.producer.mesh.get_wid() -> index
  func.func @top1.producer(%arg0: memref<16x16xi32>) attributes {itypes = "s", otypes = ""} {
    %0 = call @top1.producer.mesh.get_wid() : () -> index
    %1 = affine.load %arg0[0, 0] : memref<16x16xi32>
    allo.put_stream_global %1, @top1.pipe[] : i32 // put value to the stream
    return
  }
  func.func private @top1.consumer.mesh.get_wid() -> index
  func.func @top1.consumer(%arg0: memref<16x16xi32>) attributes {itypes = "s", otypes = ""} {
    %0 = call @top1.consumer.mesh.get_wid() : () -> index
    %1 = allo.get_stream_global @top1.pipe[] {signed} : i32 // get value from the stream
    affine.store %1, %arg0[0, 0] : memref<16x16xi32>
    return
  }
}
```

A `[16, 16]` stream array where each slot is accessed by index; `meta_for` unrolls into `loop_type = "unroll"` affine loops:

```python
@spmw.unit()
def top(A: int32[16, 16], B: int32[16, 16]):
    pipe: Stream[int32][16, 16] # stream array, each element is a scaler stream

    @spmw.work(grid=[1])
    def producer():
        with allo.meta_for(16) as i:
            with allo.meta_for(16) as j:
                pipe[i, j].put(A[i, j])

    @spmw.work(grid=[1])
    def consumer():
        with allo.meta_for(16) as i:
            with allo.meta_for(16) as j:
                B[i, j] = pipe[i, j].get()
```

```mlir
module {
  func.func @top(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) attributes {itypes = "ss", otypes = ""} {
    allo.grid_map() sharding = [] grid = [1] {
      func.call @top.producer(%arg0) : (memref<16x16xi32>) -> ()
    } : 
    allo.grid_map() sharding = [] grid = [1] {
      func.call @top.consumer(%arg1) : (memref<16x16xi32>) -> ()
    } : 
    return
  }
  allo.stream_global @top.pipe : !allo.stream<i32, 2> [16, 16] {_stream} // stream array declared as a global symbol
  func.func private @top.producer.mesh.get_wid() -> index
  func.func @top.producer(%arg0: memref<16x16xi32>) attributes {itypes = "s", otypes = ""} {
    %0 = call @top.producer.mesh.get_wid() : () -> index
    affine.for %arg1 = 0 to 16 {
      affine.for %arg2 = 0 to 16 {
        %1 = affine.load %arg0[%arg1, %arg2] : memref<16x16xi32>
        allo.put_stream_global %1, @top.pipe[%arg1, %arg2] : i32 // access stream with index
      } {loop_type = "unroll"} // compile time unrolled loop
    } {loop_type = "unroll"}
    return
  }
  func.func private @top.consumer.mesh.get_wid() -> index
  func.func @top.consumer(%arg0: memref<16x16xi32>) attributes {itypes = "s", otypes = ""} {
    %0 = call @top.consumer.mesh.get_wid() : () -> index
    affine.for %arg1 = 0 to 16 {
      affine.for %arg2 = 0 to 16 {
        %1 = allo.get_stream_global @top.pipe[%arg1, %arg2] {signed} : i32 // access stream with index
        affine.store %1, %arg0[%arg1, %arg2] : memref<16x16xi32> 
      } {loop_type = "unroll"}
    } {loop_type = "unroll"}
    return
  }
}
```

### Hierarchical Design

A `spmw.unit` can be invoked from inside another unit's worker. The callee becomes a `func.func` with its own `allo.grid_map`, and the caller's worker issues a plain `func.call` to it.

```python
@spmw.unit()
def vadd(A: int32[1024], B: int32[1024]):
    @spmw.work(grid=[1])
    def core():
        for i in allo.grid(1024):
            B[i] = A[i] + 1

@spmw.unit()
def top(A: int32[1024], B: int32[1024]):
    @spmw.work(grid=[1])
    def core():
        vadd(A, B)
```

```mlir
module {
  func.func @top(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {itypes = "ss", otypes = ""} {
    allo.grid_map() sharding = [] grid = [1] {
      func.call @top.core(%arg0, %arg1) : (memref<1024xi32>, memref<1024xi32>) -> ()
    } : 
    return
  }
  func.func private @top.core.mesh.get_wid() -> index
  func.func @top.core(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {itypes = "ss", otypes = ""} {
    %0 = call @top.core.mesh.get_wid() : () -> index
    call @vadd(%arg0, %arg1) : (memref<1024xi32>, memref<1024xi32>) -> ()  // nested unit call, lowered to a plain func.call
    return
  }
  // The callee unit
  func.func @vadd(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {itypes = "ss", otypes = ""} {
    allo.grid_map() sharding = [] grid = [1] {  // callee unit keeps its own grid_map
      func.call @vadd.core(%arg0, %arg1) : (memref<1024xi32>, memref<1024xi32>) -> ()
    } : 
    return
  }
  func.func private @vadd.core.mesh.get_wid() -> index
  func.func @vadd.core(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {itypes = "ss", otypes = ""} {
    %0 = call @vadd.core.mesh.get_wid() : () -> index
    affine.for %arg2 = 0 to 1024 {
      %1 = affine.load %arg0[%arg2] : memref<1024xi32>
      %2 = arith.extsi %1 {signed} : i32 to i33
      %c1_i33 = arith.constant 1 : i33
      %3 = arith.addi %2, %c1_i33 {signed} : i33
      %4 = arith.trunci %3 {signed} : i33 to i32
      affine.store %4, %arg1[%arg2] : memref<1024xi32>
    } {loop_type = "grid"}
    return
  }
}
```

The same callee can be invoked multiple times from one worker. Only one `@vadd` definition is emitted; the two call sites differ only in their argument bindings.

```python
@spmw.unit()
def vadd(A: int32[1024], B: int32[1024]):
    @spmw.work(grid=[1])
    def core():
        for i in allo.grid(1024):
            B[i] = A[i] + 1

@spmw.unit()
def top(A0: int32[1024], A1: int32[1024], B: int32[1024], C: int32[1024]):
    @spmw.work(grid=[1])
    def core():
        vadd(A0, B)
        vadd(A1, C)
```

```mlir
module {
  func.func @top(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>) attributes {itypes = "ssss", otypes = ""} {
    allo.grid_map() sharding = [] grid = [1] {
      func.call @top.core(%arg0, %arg2, %arg1, %arg3) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    } : 
    return
  }
  func.func private @top.core.mesh.get_wid() -> index
  func.func @top.core(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>) attributes {itypes = "ssss", otypes = ""} {
    %0 = call @top.core.mesh.get_wid() : () -> index
    call @vadd(%arg0, %arg1) : (memref<1024xi32>, memref<1024xi32>) -> ()  // two call sites share one @vadd
    call @vadd(%arg2, %arg3) : (memref<1024xi32>, memref<1024xi32>) -> ()
    return
  }
  func.func @vadd(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {itypes = "ss", otypes = ""} {
    allo.grid_map() sharding = [] grid = [1] {
      func.call @vadd.core(%arg0, %arg1) : (memref<1024xi32>, memref<1024xi32>) -> ()
    } : 
    return
  }
  func.func private @vadd.core.mesh.get_wid() -> index
  func.func @vadd.core(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {itypes = "ss", otypes = ""} {
    %0 = call @vadd.core.mesh.get_wid() : () -> index
    affine.for %arg2 = 0 to 1024 {
      %1 = affine.load %arg0[%arg2] : memref<1024xi32>
      %2 = arith.extsi %1 {signed} : i32 to i33
      %c1_i33 = arith.constant 1 : i33
      %3 = arith.addi %2, %c1_i33 {signed} : i33
      %4 = arith.trunci %3 {signed} : i33 to i32
      affine.store %4, %arg1[%arg2] : memref<1024xi32>
    } {loop_type = "grid"}
    return
  }
}
```

A wrapper worker can dispatch different hierarchical calls based on its worker ID. The `x.id == 0` branch lowers to an `scf.if` over the `get_wid` result, and each branch calls the same `@inner` unit with a different output buffer:

```python
M, N, K = 32, 32, 32
P0, P1 = 2, 2

@spmw.unit()
def inner(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    @spmw.work(grid=[P0, P1])
    def gemm():
        x, y = spmw.axes()
        pi, pj = x.id, y.id
        Mt: ConstExpr[int32] = M // P0
        Nt: ConstExpr[int32] = N // P1
        for i in range(pi * Mt, (pi + 1) * Mt):
            for j in range(pj * Nt, (pj + 1) * Nt):
                for k in range(K):
                    C[i, j] += A[i, k] * B[k, j]

@spmw.unit()
def top(A: float32[M, K], B: float32[K, N], C1: float32[M, N], C2: float32[M, N]):
    @spmw.work(grid=[2])
    def wrapper():
        x = spmw.axes()
        if x.id == 0:
            inner(A, B, C1)
        else:
            inner(A, B, C2)
```

```mlir
#map = affine_map<()[s0] -> (s0 * 16)>
#map1 = affine_map<()[s0] -> ((s0 + 1) * 16)>
module {
  func.func @top(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>, %arg3: memref<32x32xf32>) attributes {itypes = "____", otypes = ""} {
    allo.grid_map() sharding = [] grid = [2] {
      func.call @top.wrapper(%arg0, %arg1, %arg2, %arg3) : (memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
    } : 
    return
  }
  func.func private @top.wrapper.mesh.get_wid() -> index
  func.func @top.wrapper(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>, %arg3: memref<32x32xf32>) attributes {itypes = "____", otypes = ""} {
    %0 = call @top.wrapper.mesh.get_wid() : () -> index
    %1 = arith.index_cast %0 {unsigned} : index to i32  // x.id == 0 branch, lowered to scf.if
    %c0_i32 = arith.constant 0 : i32
    %2 = arith.cmpi eq, %1, %c0_i32 : i32
    scf.if %2 {
      func.call @inner(%arg0, %arg1, %arg2) : (memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
    } else {
      func.call @inner(%arg0, %arg1, %arg3) : (memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
    }
    return
  }
  func.func @inner(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) attributes {itypes = "___", otypes = ""} {
    allo.grid_map() sharding = [] grid = [2, 2] {
      func.call @inner.gemm(%arg0, %arg1, %arg2) : (memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
    } : 
    return
  }
  func.func private @inner.gemm.mesh.get_wid() -> (index, index)
  func.func @inner.gemm(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) attributes {itypes = "___", otypes = ""} {
    %0:2 = call @inner.gemm.mesh.get_wid() : () -> (index, index)
    affine.for %arg3 = #map()[%0#0] to #map1()[%0#0] {
      affine.for %arg4 = #map()[%0#1] to #map1()[%0#1] {
        affine.for %arg5 = 0 to 32 {
          %1 = affine.load %arg2[%arg3, %arg4] : memref<32x32xf32>
          %2 = affine.load %arg0[%arg3, %arg5] : memref<32x32xf32>
          %3 = affine.load %arg1[%arg5, %arg4] : memref<32x32xf32>
          %4 = arith.mulf %2, %3 {_float} : f32
          %5 = arith.addf %1, %4 {_float} : f32
          affine.store %5, %arg2[%arg3, %arg4] : memref<32x32xf32>
        }
      }
    }
    return
  }
}
```