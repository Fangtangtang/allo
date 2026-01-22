# Analysis of `allo/ir/infer.py`

## Current Implementation Overview

`allo/ir/infer.py` defines the `TypeInferer` class, which traverses the Python AST to infer types and shapes for Allo kernels. It is a critical component for the frontend, ensuring that Python code is correctly typed before lowering to MLIR.

## Identified Issues

### 1. Monolithic Visitor Methods
Some visitor methods are excessively long and handle multiple distinct responsibilities.
- **`visit_FunctionDef`**: This method handles:
    - Nested function definitions.
    - `@df.kernel` decorator processing (`mapping`, `args`).
    - Unrolling logic (`ctx.unroll`).
    - Predicate generation for execution masks.
    - Scope management.
    - Recursion for kernel instantiation.
  This makes it hard to read, maintain, and test.

- **`visit_type_hint`**: This method parses type hints and handles:
    - `Subscript` (e.g., `int32[32]`, `Stream[T, N]`).
    - `Name` (e.g., `int32`).
    - `Call` (e.g., `Fixed(16, 8)`).
    - `Constant` (string hints).
    - `BinOp` (layout/memory refinement `Ty[M] @ stateful`).
  It should be split or delegated to specific handlers.

### 2. Overloaded Methods
- **`visit_Subscript`**: This method handles three distinct operations:
    - **Struct Field Access**: `struct['field']`.
    - **Tensor Slicing/Indexing**: `A[i, j]`.
    - **Bit Operations**: `int_val[0:4]`.
  Mixing these logics leads to complex conditional branches and makes it fragile.

### 3. Hardcoded String Checks
The code relies on checking hardcoded strings for certain built-ins and attributes:
- `get_pid`, `range`, `grid`, `reduction`.
- `node.attr == "T"`, `reverse`, `copy`, `bits`, `fracs`.
This reduces flexibility and makes rename/refactor of those built-ins harder.

### 4. Global State Reliance
The `TypeInferer` relies heavily on `ctx.global_vars`. While necessary for symbol resolution, the direct manipulation and deep recursion with copied contexts in `visit_FunctionDef` can be error-prone.

## Refactoring Plan

### 1. Split `visit_FunctionDef`
- Extract `@df.kernel` handling into a separate helper method.
- Extract unrolling and predicate generation logic.

### 2. Decompose `visit_type_hint`
- Create specific handlers for different hint types (Stream, ConstExpr, Arrays).

### 3. Refactor `visit_Subscript`
- Check the type of the `value` being subscripted first (Struct vs Tensor vs Scalar).
- Dispatch to specialized methods: `visit_struct_access`, `visit_tensor_access`, `visit_bit_access`.

### 4. Modularize Built-in Handling
- Instead of raw string checks in `visit_Call` or `visit_Attribute`, consider a dispatch mechanism or a more robust registry of supported built-ins.
