# Allo Language Specification

This document defines the specification for Allo source programs. Allo is a domain-specific language for accelerator design, embedded in Python.

## Part 1: Basic Allo

Basic Allo covers the core language features for defining hardware kernels, including types, control flow, and operations. Basic Allo programs are typically defined within a function decorated with `allo.customize` (conceptually) or intended to be passed to it.

### 1. Type System

Allo supports a rich set of hardware-aware types found in `allo.ir.types`.

#### Primitive Types
*   **Integer**: `Int(width)`, `UInt(width)`
    *   Aliases: `int8`, `int16`, `int32`, `int64` (Signed)
    *   `uint1`, `uint8`, `uint16`, `uint32` (Unsigned)
    *   `index` (Platform dependent, typically for loops)
*   **Fixed Point**: `Fixed(width, frac)`, `UFixed(width, frac)`
*   **Floating Point**: `Float(width, exponent_width)`
    *   Aliases: `float16`, `float32`, `float64`, `bfloat16`
*   **Boolean**: `bool` (1-bit)

#### Special Types
*   **ConstExpr**: Compile-time constant.
    *   Syntax: `name: ConstExpr[Type] = value`
    *   Usage: Used for values that must be resolved at compile time (e.g., offsets, loop bounds).
    *   Example: `offset: ConstExpr[int32] = 2`
*   **Stateful Variables**: Variables that persist value across kernel invocations (static variables).
    *   Syntax: `name: Type @ stateful = initial_value`
    *   Example: `acc: int32 @ stateful = 0`

#### Composite Types
*   **Arrays (MemRefs)**: Represented as typed memory regions.
    *   Syntax: `Type[Shape]`
    *   Example: `int32[32, 32]`, `float32[10]`
*   **Structs**: Custom named fields.
    *   Definition: `allo.ir.types.Struct({"field": Type, ...})`

#### Type Representation in Python
*   Use type hints in function signatures and variable declarations.
*   Example: `def kernel(A: int32[32]) -> int32:`

### 2. Program Structure

#### Kernel Definition
A kernel is a Python function. It serves as the entry point for hardware generation.

```python
def kernel(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
    C: int32[32, 32] = 0
    # ... logic ...
    return C

def scalar_op(a: int32, b: int32) -> int32:
    return a + b
```

#### Polymorphism
Kernels can be generic using Type Variables.

```python
def kernel[T, M, N](A: "T[M, N]") -> "T[M, N]":
    # ...
```
*   `T`: Type variable
*   `M`, `N`: Integer constants (resolution time)

### 3. Control Flow

#### Loops
*   **`range(start, stop, step)`**: Standard serial loops.
    *   Unrolled by default if small, otherwise pipelined depending on backend.
*   **`allo.grid(dim1, dim2, ...)`**: Imperfectly nested loops, often used for iteration over multi-dimensional arrays.
    *   Example: `for i, j in allo.grid(32, 32):`
*   **`allo.reduction(dim1, dim2, ...)`**: Reduction loops, hinting usually for accumulation.
*   **Unsupported**: `break` and `continue` statements are currently not supported.

#### Conditionals
*   **`if` / `elif` / `else`**: Standard Python conditionals.
    *   For runtime conditions (data-dependent), generates hardware muxes/branching.
    *   For elaboration-time constants, evaluated during Trace.

### 4. Operations

*   **Arithmetic**: `+`, `-`, `*`, `/` (div), `//` (floor div), `%` (mod), `**` (pow).
*   **Bitwise**: `<<`, `>>`, `|`, `&`, `^`, `~`.
*   **Comparison**: `==`, `!=`, `<`, `<=`, `>`, `>=`.
*   **Logical**: `and`, `or`, `not`.
*   **Casting**: `Type(value)` or `float(x)`, `int(x)`. Explicit casting recommended for mixed-precision.
*   **Select**: Python ternary `a if cond else b`.

---

## Part 2: Dataflow Allo

Dataflow Allo extends basic Allo for building composed systems of kernels communicating via streams, targeting spatial architectures.

### 1. Region and Composition

*   **`@df.region()`**: Decorator for the top-level function defining the dataflow graph.
    
    ```python
    @df.region()
    def top(A: int32[16, 16]):
        # Stream definitions
        # Kernel calls
    ```

*   **`@df.kernel(...)`**: Decorator for leaf kernels within a region.
    *   `mapping=[dims...]`: Maps the kernel to a spatial grid (e.g., Systolic Array or AIE dims).
    *   `args=[...]`: Arguments passed to the kernel.
        *   The element at index `i` in the `args` list is passed as the `i`-th argument to the kernel function.
        *   Example:
            ```python
            @df.region()
            def top(A: int32[32, 32], B: int32[32, 32]):
                # args=[A, B] means:
                #   Index 0 in list (A) -> 1st arg of 'core' (local_A)
                #   Index 1 in list (B) -> 2nd arg of 'core' (local_B)
                @df.kernel(mapping=[...], args=[A, B])
                def core(local_A: int32[32, 32], local_B: int32[32, 32]):
                    pass
            ```


### 2. Communication

*   **`Stream` Type**: `Stream[Type, Depth]`
    *   Must be instantiated inside the region but outside kernels.
*   **Operations**:
    *   `stream.put(value)`: Non-blocking write (semantics depend on FIFO full).
    *   `stream.get()`: Blocking read.

### 3. Memory Layout Refinement

Used to specify how data is distributed across spatial resources.

*   **Import**: `from allo.memory import Layout`
*   **Types**:
    *   `Layout.Shard(dim)`: Shard the data along the spatial dimension specified by `dim`.
        *   `dim` refers to the index of the `mapping` list defined in the `@df.kernel(...)` decorator.
    *   `Layout.Replicate`: Replicate the data across the spatial domain.
*   **Syntax**: `Type @ Layout` list.
    *   The layout list elements are applied to the corresponding dimensions of the data's shape.
    *   Example: `local_A: Ty[M, K] @ [Layout.Shard(1), Layout.Replicate]`
        *   `Layout.Shard(1)` is applied to the first dimension `M`.
        *   `Layout.Replicate` is applied to the second dimension `K`.

### 4. Metaprogramming in Dataflow

For defining parametric connectivity and composed logic.

*   **`allo.meta_if(cond)` / `meta_else()` / `meta_elif(cond)`**:
    *   Generates static structure based on elaboration-time constants (like PID).
    *   **Constraint**: The condition must be a compile-time constant.
*   **`allo.meta_for`**:
    *   Unrolls logic at compile time.
    *   **Constraint**: Loop bounds and steps must be compile-time constants.
*   **`df.get_pid()`**:
    *   Returns the coordinate of the current kernel instance in the spatial map.

---

## Part 3: Language Semantics

### 1. Variable Scoping (Block Scope)

Allo enforces C++-style **Block Scoping** rules, which differs from standard Python (function scoping).

*   **Scope Boundaries**: `if`, `elif`, `else`, `for`, `while`, `meta_if`, `meta_for`, `meta_else`.
*   **Rule**: A variable declared for the first time inside a block is **local** to that block. It is not visible after the block exits.
*   **Access**: Inner blocks can read/write variables defined in outer blocks.

### 2. Variable Declaration

*   **Explicit Declaration**: It is best practice (and often required for correctness in IR generation) to strictly type variables upon first assignment using type hints.
    *   `v: int32 = 0`
*   **Reassignment Validity**:
    *   A variable can be reassigned.
    *   The new value must match the declared type of the variable.
    *   **Immutable Constants**: `ConstExpr` variables and values returned by `df.get_pid()` are compile-time constants and **cannot** be reassigned.

---
