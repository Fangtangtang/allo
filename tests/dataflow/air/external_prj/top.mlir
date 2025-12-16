module {
  air.channel @ChanIn []
  air.channel @ChanOut []
  func.func private @passThroughLine(memref<1024xui8, 2 : i32>, memref<1024xui8, 2 : i32>, i32)
  func.func @copy(%arg0: memref<4096xui8>, %arg1: memref<4096xui8>) {
    air.launch () in () args(%arg2=%arg0, %arg3=%arg1) : memref<4096xui8>, memref<4096xui8> {
      air.channel.put  @ChanIn[] (%arg2[] [] []) : (memref<4096xui8>)
      air.channel.get  @ChanOut[] (%arg3[] [] []) : (memref<4096xui8>)
      air.segment @seg  {
        %c1 = arith.constant 1 : index
        %c1_0 = arith.constant 1 : index
        air.herd @copyherd  tile (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1_0) attributes {link_with = "passThrough.cc.o"} {
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_1 = arith.constant 1 : index
          scf.for %arg8 = %c0 to %c4 step %c1_1 {
            %alloc = memref.alloc() : memref<1024xui8, 2 : i32>
            %alloc_2 = memref.alloc() : memref<1024xui8, 2 : i32>
            air.channel.get  @ChanIn[] (%alloc[] [] []) : (memref<1024xui8, 2 : i32>)
            %c1024_i32 = arith.constant 1024 : i32
            func.call @passThroughLine(%alloc, %alloc_2, %c1024_i32) : (memref<1024xui8, 2 : i32>, memref<1024xui8, 2 : i32>, i32) -> ()
            air.channel.put  @ChanOut[] (%alloc_2[] [] []) : (memref<1024xui8, 2 : i32>)
            memref.dealloc %alloc : memref<1024xui8, 2 : i32>
            memref.dealloc %alloc_2 : memref<1024xui8, 2 : i32>
          }
        }
      }
    }
    return
  }
}

