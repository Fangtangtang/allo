def StreamGlobalOp : Allo_Op<"stream_global", [Symbol]> {
  let summary = "Create a global stream object";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$element_type, DenseI64ArrayAttr:$shape);
  let assemblyFormat = [{
      $sym_name `:` $element_type $shape  attr-dict
  }];
  let hasVerifier = 1;
}

def GlobalStreamGetOp : Allo_Op<"get_stream_global"> {
  let summary = "Get an object from a global stream";
  // arguments:
  // - $global: symbol name of the global stream
  // - $indices: variadic index operands
  // - $map: affine map
  let arguments = (
    ins FlatSymbolRefAttr:$global,
        Variadic<Index>:$indices,
        AffineMapAttr:$map
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    void simplifyAffineMap();
  }];
}

def GlobalStreamPutOp : Allo_Op<"put_stream_global"> {
  let summary = "Put an object to a global stream";
  // arguments:
  // - $global: symbol name of the global stream
  // - $indices: variadic index operands
  // - $data: value to put
  // - $map: affine map to compute indices
  let arguments = (
      ins FlatSymbolRefAttr:$global,
          Variadic<Index>:$indices,
          AnyType:$data,
          AffineMapAttr:$map
  );
  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    void simplifyAffineMap();
  }];
}


def GridMapOp : Allo_Op<"grid_map",
  [RecursiveMemoryEffects, SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Grid map operation";
  let description = [{
    The grid_map operation distributes a computation over a logical grid.
    It takes a list of memrefs with static shape (`tensors`) as input arguments.
    It contains a single block whose arguments are the sharded memrefs. The operations in the block can access variables in parent regions.
  }];

  let arguments = (ins
    Variadic<AnyStaticShapeMemRef>:$tensors,
    ArrayAttr:$sharding,
    DenseI64ArrayAttr:$grid
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `(` $tensors `)`
    `sharding` `=` $sharding
    `grid` `=` $grid
    $body
    attr-dict
    `:` type($tensors)
  }];

  let hasVerifier = 1;
}