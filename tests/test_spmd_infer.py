# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import pytest
import allo
from allo.ir.types import int32, float32, Index, ConstExpr, Stream, index
from allo.ir.spmd_infer import SPMDTypingInferer
from allo.ir.visitor import ASTContext, ASTVisitor
import allo.dataflow as df
import ast
import inspect
import textwrap
from allo._mlir.ir import Context

def infer(func):
    src, starting_line_no = inspect.getsourcelines(func)
    src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
    src = textwrap.dedent("\n".join(src))
    tree = ast.parse(src)
    with Context() as mlir_ctx:
        ctx = ASTContext(tree, globals(), mlir_ctx, verbose=True)
        visitor = SPMDTypingInferer()
        visitor.visit_FunctionDef(ctx, tree.body[0])
    return ctx

def test_basic_kernel_valid():
    @df.kernel(mapping=[4])
    def kernel(A: int32[16]):
        pass

    ctx = infer(kernel)
    assert ctx.top_func is not None

def test_get_pid_constant():
    @df.kernel(mapping=[4])
    def kernel():
        pid = df.get_pid()
        # Should be Index type and constexpr
        
    ctx = infer(kernel)
    # We need to inspect the inferred types in the body
    # The visit logic doesn't expose locals easily without inspecting ctx.scopes or modifying visitor to store it
    # But we can check if it didn't crash.
    # To really verify, let's try to reassign it

def test_get_pid_reassignment_fail():
    @df.kernel(mapping=[4])
    def kernel():
        pid = df.get_pid()
        pid = (1, 1, 1) # Should fail

    with pytest.raises(SystemExit):
        infer(kernel)

def test_constexpr_reassignment_fail():
    @df.kernel(mapping=[4])
    def kernel():
        C: ConstExpr[int32] = 10
        C = 20 # Should fail

    with pytest.raises(SystemExit):
        infer(kernel)

def test_meta_if_constant():
    @df.kernel(mapping=[4])
    def kernel():
        with allo.meta_if(1 < 2): # Constant true
            pass

    infer(kernel)

def test_meta_if_non_constant_fail():
    @df.kernel(mapping=[4])
    def kernel(A: int32[1]):
        with allo.meta_if(A[0] < 2): # Non-constant
            pass

    with pytest.raises(SystemExit):
        infer(kernel)

def test_meta_for_constant():
    @df.kernel(mapping=[4])
    def kernel():
        with allo.meta_for(10) as i:
            pass

    infer(kernel)

def test_meta_for_non_constant_fail():
    @df.kernel(mapping=[4])
    def kernel(A: int32[1]):
        with allo.meta_for(A[0]) as i: # Non-constant bound
            pass

    with pytest.raises(SystemExit):
        infer(kernel)

def test_meta_for_loop_var_const():
    @df.kernel(mapping=[4])
    def kernel():
        with allo.meta_for(10) as i:
            i = 5 # Should fail reassignment if we marked it const

    with pytest.raises(SystemExit):
        infer(kernel)

# --- Ported from tests/dataflow/test_builder.py ---

M, N = 16, 16

def test_dataflow_meta_if():
    @df.kernel(mapping=[1])
    def kernel(local_A: int32[M, N], local_B: int32[M, N]):
        pid = df.get_pid()
        with allo.meta_if(pid > 0):
            local_B[:, :] = local_A
        with allo.meta_else():
            local_B[:, :] = local_A

    ctx = infer(kernel)
    # Basic check to see if it runs through

def test_stream_inference():
    # Mocking usage of stream in a kernel. 
    # Since Stream definition is usually outside kernel in region, 
    # we might need to assume proper context or just check Inside-Kernel usage logic 
    # where Stream is passed as arg or available in global.
    # In SPMD infer, we often check kernels. 
    
    # Let's define a kernel that uses a stream argument
    @df.kernel(mapping=[1])
    def kernel(pipe: Stream[int32, 4], A: int32[M, N]):
        pid = df.get_pid()
        # pipe.put(A) # broad cast issue maybe? pipe expects Ty?
        # A is [16, 16], pipe is Stream[int32]
        # pipe.put expects int32. 
        # let's try reading
        val = pipe.get()
        # val should be int32

    ctx = infer(kernel)
    # Check if pipe.get() return type is inferred correctly
    # We can't easily check local var types without inspecting ctx or visitor side effects.
    # But if it didn't crash, good sign.

# --- Ported from tests/test_builder.py ---

def test_standard_ops_integers():
    @df.kernel(mapping=[1])
    def kernel():
        a: int32 = 10
        b: int32 = 20
        c = a + b
        d = a * b
        e = a - b
        f = a // b
        g = a % b
        
    infer(kernel)

def test_standard_ops_floats():
    @df.kernel(mapping=[1])
    def kernel():
        a: float32 = 10.0
        b: float32 = 20.0
        c = a + b
        d = a * b
        e = a / b
        
    infer(kernel)

def test_bitwise_ops():
    @df.kernel(mapping=[1])
    def kernel():
        a: int32 = 10
        b: int32 = 20
        c = a & b
        d = a | b
        e = a ^ b
        f = a << 2
        g = b >> 1
        
    infer(kernel)

def test_comparison_ops():
    @df.kernel(mapping=[1])
    def kernel():
        a: int32 = 10
        b: int32 = 20
        c = a < b
        d = a > b
        e = a == b
        f = a != b
        
    infer(kernel)

def test_linalg_matmul_infer():
    @df.kernel(mapping=[1])
    def kernel(A: int32[32, 32], B: int32[32, 32]):
        C = allo.matmul(A, B)
        # C should be int32[32, 32]
        
    ctx = infer(kernel)

def test_nested_if_logic():
    @df.kernel(mapping=[1])
    def kernel(a: int32, b: int32):
        r: int32 = 0
        if a == 0:
            r = 1
        elif a == 1:
            r = 2
            if b == 2:
                r = 3
        else:
            r = 4
            
    infer(kernel)

def test_while_loop():
    @df.kernel(mapping=[1])
    def kernel(A: int32[10]):
        i: index = 0
        while i < 10:
            A[i] = i
            i += 1
            
    infer(kernel)

def test_select_ternary():
    @df.kernel(mapping=[1])
    def kernel(A: int32[32], B: int32[32]):
        for i in range(32):
            B[i] = 1 if A[i] % 2 == 0 else 0
            
    infer(kernel)

def test_broadcast_shapes():
    @df.kernel(mapping=[1])
    def kernel(A: int32[32, 1], B: int32[1, 32]):
        C = A + B # Should result in [32, 32]
        
    infer(kernel)

def test_subscript_slicing():
    @df.kernel(mapping=[1])
    def kernel(A: int32[10, 10]):
        B = A[0:5, :] # [5, 10]
        C = A[0, :] # [10]
        
    infer(kernel)


if __name__ == "__main__":
    pytest.main([__file__])
