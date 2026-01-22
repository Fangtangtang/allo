We wannt a seperate infer and IR builder to construct Allo IR retaining the high-level semantics of the original code. This will retain the structure of

1. SPMD style kernels. Avoid unrolling the instances on the grid in high-level IR. 
2. Keep meta-programming features of the original code. This includes meta_if/meta_elif/meta_else and meta_for.

I'd like to write a spmw_infer.py mirroring infer.py and spmw_builder.py mirroring builder.py for building this high-level IR.
This implementation will be a refactoring of the existing infer.py and builder.py files. I'd like to remove the flags (e.g. ctx.unroll) used in the original code, avoid using some attributes in `ctx` and keep the code logic simple and readable.
