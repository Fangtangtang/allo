We wannt a seperate infer and IR builder to construct Allo IR retaining the high-level semantics of the original code. This will retain the structure of

1. SPMD style kernels. Avoid unrolling the instances on the grid in high-level IR. 
2. Keep meta-programming features of the original code. This includes meta_if/meta_elif/meta_else and meta_for.

I'd like to write a spmw_infer.py mirroring infer.py and spmw_builder.py mirroring builder.py for building this high-level IR.
This implementation will be a refactoring of the existing infer.py and builder.py files. I'd like to remove the flags (e.g. ctx.unroll) used in the original code, avoid using some attributes in `ctx` and keep the code logic simple and readable.

## Step 1
Implement spmv_infer.py based on infer.py.

Some specific requirements:

1. Do not use things like ctx.unroll, ctx.symbolic. You should try your best to use less attributes in `ctx`.
2. Treat the results of `get_pid` as constants. (maybe reuse some logic of ConstExpr).
3. I need to do validity checking for 'reassignment of variables'. e.g., reassigning constant variables.
4. Keep meta-programming features of the original code.
    - check whether the condition of meta_if is a compile time constant expression.
    - check whether the loop arguments of meta_for is a compile time constant expression.
    - Only visit the meta_for loop body once.
    - Only visit one kernel once, so do not unroll the `*mapping`.

