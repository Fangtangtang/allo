<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->
# Frontend Syntax

This document provides a comprehensive reference for the Allo frontend syntax and semantics.
Allo uses a Python-based domain-specific language (DSL) that requires **strict type annotations** to enable hardware synthesis and optimization.

*🔀 for dataflow programming

## Data Types

### Base Data Types
Currently, Allo supports four base data types.

#### `Index`
An array index.

#### Integers
* `Int`: An integer of variable bitwidth. `Int(bits)`
* `UInt`: An unsigned integer of variable bitwidth. `UInt(bits)`

#### Floating points
A floating point decimal number.

| Format        | Bits | Exponent | Fraction |
|---------------|------|----------|----------|
| float64(FP64) | 64   | 11       | 52       |
| float32(FP32) | 32   | 8        | 23       |
| float16(FP16) | 16   | 5        | 10       |
| bfloat16(BF16)| 16   | 8        | 7        |


#### Fixed points
* `Fixed`: A fixed point decimal. `Fixed(bits, fracs)`
* `UFixed`: An unsigned fixed point decimal. `UFixed(bits, fracs)`

#### Type Casting

* [Implicit Casting](https://cornell-zhang.github.io/allo/gallery/dive_01_data_types.html#implicit-casting)

* [Explicit Casting](https://cornell-zhang.github.io/allo/gallery/dive_01_data_types.html#explicit-casting)

### Derived Types
Derived types are constructed from base data types. They extend base types with additional structural or semantic properties.

#### Tensor Types
A tensor type is constructed by attaching [**compile-time**](#compile-time-constants) shape information
to a [base data type](#base-data-types) using bracket notation:

```txt
BaseType[Dim0, Dim1, ..., DimN]
```

A tensor type represents a multi-dimensional array whose element type
is the given base data type.

For example:
```python
int32[32, 32]
```
denotes a 2-dimensional tensor of shape `(32, 32)` whose element type is `int32`.

#### `ConstExpr`
`ConstExpr[T]` is a derived type that denotes a [compile-time constant](#compile-time-constants) of base type `T`:

```txt
ConstExpr[BaseType]
```

A `ConstExpr` variable must:
* Be parameterized by a [base data type](#base-data-types).
* Be initialized at the point of declaration. Uninitialized `ConstExpr` declarations are illegal.

For example:
```python
base: ConstExpr[int32] = 2
mult: ConstExpr[int32] = 3
offset: ConstExpr[int32] = base * mult  # Computed at compile time: 6
```

#### 🔀`Stream`
`Stream[T, D]` declares a point-to-point FIFO channel for communication:

```txt
Stream[ElementType, Depth]
```
* `ElementType` is the type of elements carried by the stream. It can be either a [base data type](#base-data-types) or a [tensor type](#tensor-types).
* `Depth` (optional) is the FIFO depth of the stream. If omitted, the default depth is 2.

For example:

```python
# Basic scalar stream
pipe: Stream[float32, 4]  # Stream of float32 with depth 4
# Stream of unsigned integers
stream: Stream[UInt(16), 4]
# Default depth (2)
default_pipe: Stream[int32] # Stream of int32 with depth 2
```

##### Stream Arrays
A stream array is constructed by attaching shape information to a stream type using bracket notation:
```txt
Stream[T, D][Dim0, Dim1, ..., DimN]
```
This declares a multi-dimensional array of independent FIFO channels.

For example:
```python
# 2D array of streams
fifo_A: Stream[float32, 4][4, 4]
```
### Refinement Types
Refinement types extend an existing type, including [base type](#base-data-types) and [derived type](#derived-types), by attaching additional attributes using the following syntax:
```txt
Type @ Refinement
```

## Function Definition 

### Function Signature
Allo functions are defined as Python functions with explicit type annotations for all arguments and return types.
Arguments and return types can be [scalar](#base-data-types) or [tensor types](#tensor-types). The syntax follows Python's type hint notation:

```python
def kernel(arg1: Type1[Shape1], arg2: ScalarType) -> ReturnType[Shape]:
    # function body
    return result

```
**Example: Scalar Arguments**

```python
def kernel(A: int32) -> int32:
    return A + 1
```

**Example: Matrix Multiplication**

```python
from allo.ir.types import int32

def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
    C: int32[32, 32] = 0
    for i, j, k in allo.grid(32, 32, 32):
        C[i, j] += A[i, k] * B[k, j]
    return C

```
#### Argument Semantics
Allo distinguishes between scalar and tensor arguments in terms of passing semantics and mutability:

* Scalar arguments
  * Passed by value.
  * Treated as **read-only** inside the function body.
  * Reassigning a scalar argument inside the function results in undefined behavior.

* Tensor arguments
  * Passed by reference.
  * Can be both **read and written** inside the function.
  * Modifications to tensor arguments are visible to the caller.

#### Multiple Return Values

Functions can return multiple values as a tuple:

```python
def kernel(A: int32[M], B: int32[M]) -> (int32[M], int32[M]):
    res0: int32[M] = 0
    res1: int32[M] = 0
    for i in range(M):
        res0[i] = A[i] + 1
        res1[i] = B[i] + 1
    return res0, res1
```
The caller can unpack the returned tuple:

```python
C, D = callee(A[i], B[i])

```
To ignore certain return values, use underscore:

```python
C, _ = callee(A[0], B[0])  # Ignore second return value


```
#### No Return Value

Functions that don't return a value can omit the return type annotation, use ``-> None``,
or have an empty return:

```python
def kernel(A: int32[32]):
    pass  # No return

def kernel(A: int32[32]) -> None:
    return

def kernel(A: int32[32]):
    return None
```
## Variables
Different from native Python, Allo requires the program to be strongly and statically typed.
Variables are declared using Python's type annotation syntax:
```txt
var: Type = init_value
```
* `Type` can be [base type](#base-data-types) or [derived type](#derived-types) with supported [efinement type](#refinement-types).
* `init_value` is optional. 

For example:
```python
# Declaration with initialization
x: int32 = 0
# Declaration without initialization
y: int32
# Assignment after declaration
z = x + y
# Declaration for 1D tensor
A: int32[10] = 0
```

### Assignment Semantics

[Assignments](#assignment-statements) in Allo follow **value-copy semantics**.

An assignment copies the value of the right-hand side expression into the target variable. It **does not** transfer object ownership or create reference aliases.

For example:
```python
temp: int32[4, 4] = 0
outp: int32[4, 4] = temp  # Copy temp to outp

# Copy from argument
def kernel(inp: int32[8, 8]) -> int32[8, 8]:
    outp: int32[8, 8] = inp
    return outp
```

Type Compatibility:
- The value being assigned must be type-compatible with the target variable.
- For [tensor types](#tensor-types), broadcasting may be implicitly performed if required and valid.
   - For example:
        ```python
        A: int32[4] = 0     # Scalar broadcast to tensor
        ```

- For [base types](#base-data-types), implicit casting is allowed when it is safe and well-defined.
   - For example:
        ```python
        x: int32 = 0
        y: float32 = x      # Implicit cast if allowed
        ```
- `ConstExpr`s are **immutable** and cannot be reassigned.

### Variable Scoping
Allo enforces C++-style **Block Scoping** rules, which differs from standard Python.

*   **Scope Boundaries**: branch, loop body.
*   **Rule**: A variable declared for the first time inside a block is **local** to that block. It is not visible after the block exits. Inner blocks can read/write variables defined in outer blocks.

Variables should be declared in the scope where they are used or parenting scopes:
```python
def kernel(a: int32) -> int32:
    r: int32 = 0  # Declare outside conditional
    if a == 0:
        r = 1
    else:
        r = 4
    return r
```

Local variables are accessible within its scope:
```python
def kernel(a: int32) -> int32:
    r: int32 = 0
    if a > 0:
        t: int32 = 1  # Local to if-branch
        r = r + t
    return r
```

Allo enforces strict symbol uniqueness within a scope.
Declaring a variable with the same name more than once in the same scope results in a symbol conflict.

```python
r: int32 = 0
# annotated assignment without init_value is taken as varibale declaration
r: int32 # Error: symbol conflict
```

#### Invalid Scoping
The following patterns will raise errors:

**Using variables outside the loop:**

```python
# ERROR: tmp is not accessible outside loop
def kernel(n: int32) -> int32:
    for i in range(n):
        tmp: int32 = i
    return tmp  # Error: tmp not in scope

# ERROR: r is not accessible outside branches
def kernel(a: int32) -> int32:
    if a == 0:
        r: int32 = 1
    else:
        r: int32 = 4
    return r  # Error: r not in scope
```

**Redefining loop variables in nested loops:**

```python
# ERROR: Cannot redefine i in nested loop
def kernel(n: int32) -> int32:
    s: int32 = 0
    for i in range(n):
        for i in range(n):  # Error: i already defined
            s = s + i
    return s
```

## Expressions
### Arithmetic Expressions
#### Unary Arithmetic Operators
| Operator | Description     | Example |
|----------|-----------------|---------|
| ``+``    | Unary plus      |   `+a`  |
| ``-``    | Unary negation  |   `-a`  |

#### Binary Arithmetic Expressions
| Operator | Description     | Example |
|----------|-----------------|---------|
| ``+``    | Addition        |`a + b`  |
| ``-``    | Subtraction     | `a - b` |
| ``*``    | Multiplication  | `a * b` |
| ``/``    | Division (float)| `a / b` |
| ``//``   | Floor division  | `a // b`|
| ``%``    | Modulo          | `a % b` |

### Logic Expressions
#### Logical Operators
| Operator | Description | Example   |
| -------- | ----------- | --------- |
| `and`    | Logical AND | `a and b` |
| `or`     | Logical OR  | `a or b`  |
| `not`    | Logical NOT | `not a`   |

### Comparison Expressions
All standard comparison operators are supported: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``.

**Restriction**: Chained comparisons are currently not supported.


### Call Expressions

Allo adopts Python's function call syntax:
```python
callee(arg0, arg1, ..., argN)
```

Only positional arguments are supported. Keyword arguments, default arguments, and variadic arguments (`*args`, `**kwargs`) are not supported.

- The number of arguments must exactly match the function signature.
- Each argument must be type-compatible with its corresponding parameter.
- Implicit casting (for base types) and broadcasting (for tensor types)
  may be performed if the conversion is valid under the type rules.

## Statements
### Assignment Statements
#### Assignment
The syntax of assignment in Allo follows standard Python syntax:
```python
target = expression
```

Multiple assignment and tuple unpacking are also supported:
```python
a, b = 1, 2
x, y = foo()
```
**Restriction**: assignments where the left-hand side and right-hand side refer to overlapping variables are currently not supported.
```python
a, b = b, a   # Not supported
```

#### Augmented Assignment
Augmented assignment with operators supported in [binary arithmetic](#binary-arithmetic-expressions) work on both scalars and tensor elements.

#### Annotated Assignment
Annotated Assignment without right-hand side expression is taken as varibale declaration. Otherwise, if the target does not exist in current [scope](#variable-scoping), it's taken as declareation and initialization with right-hand side expression; if it exists, it's taken as reassignment.

A variable may be reassigned using type annotation syntax, provided that the annotated type matches the original declared type. The annotated type must be identical to the original declaration type.

For example:
```python
r: int32 = 0
# annotated assignment with init_value is taken as annotated reassignment
r: int32 = 2
```

### Conditional Statements
Allo supports Python-like conditional statements with `if`, `elif`, and `else`:

```python
if condition:
    # then-block
elif other_condition:
    # elif-block
else:
    # else-block
```

* Nested `if` statements are supported.
* The condition expression must evaluate to a boolean-compatible type.
    * If a condition is a [compile-time constant](#compile-time-constants), Allo performs Dead Code Elimination (DCE), removing the dead branch if the condition is `False`.


### Loop Statements
**Restriction**: `break` and `continue` statements are not supported.

#### For Loops
Allo supports Python-like `for` loops using `range`.

```python
for i in range(start, end, step):
    ...
```
* `start` (optional): default to 0, must evaluate to a index-compatible type.
* `end` (required) must evaluate to a index-compatible type.
* `step` (optional): default to 1, must evaluate to a index-compatible type.

If `range` is called with two arguments, they are interpreted as:
```python
range(start, end)   # start = first argument, end = second argument
```

#### While Loops
`while` loops also follow Python syntax:
```python
while condition:
    # loop body
```
The condition expression must evaluate to a boolean-compatible type.

#### Nested Loops
`allo.grid` provides a shorthand for nested loops:
```python
for i0, i1, ..., iN  in allo.grid(dim0, dim1, ..., dimN) 
```
Generates N nested loops. Loop `i_k` iterates from 0 (inclusive) to `dimk` (exclusive) with step 1. Loops are nested from outermost (`dim0`) to innermost (`dimN`).

`dimk` must evaluate to a index-compatible type.

```python
# 3D grid: Equivalent to three nested for loops
for i, j, k in allo.grid(32, 32, 32):
    C[i, j] += A[i, k] * B[k, j]

# 2D grid
for i, j in allo.grid(M, M):
    res[i, j] = C[i, j] + 1
```

### Meta-Programming
#### Meta For (Compile-Time Loop Unrolling)
<!-- TODO -->

## Built-in Functions
### 🔀Meta Functions
**Note:** reading dataflow [unit and work definitions](#programs) helps understanding.

#### `axes()`
Only used in [works](#works).

Returns a tuple of axes of the work grid.  
The result must be unpacked. Each element is an axis object that can be used in [shard](#tensorshardaxes). The coordinate of the current work along that axis is available via `.id`.

For example:
```python
x, y = spmw.axes() # axes of 2D work grid
wid: index = x.id # coordinate along axis x
``` 

#### `Tensor.shard(axes)`
Applies sharding to a global tensor `Tensor`.
This function provides a way to specify how global tensors are partitioned and mapped to [works](#works).

Only used in [works](#works) for tensor partitioning. It's recommended to use this at the beginning of a work function.

`axes` is a list whose length must match the rank of the tensor.  
Each element specifies how the corresponding tensor dimension is mapped to the work grid:

- `None`: the dimension is replicated across all works.
- `axis`: must be an axis object returned by [axes()](#axes). the dimension is sharded along the given work-grid axis.

When a dimension is sharded, its size must be evenly divisible by the number of works along that axis.

For example:
```python
@spmw.unit()
def top(A: int32[1, 1024], B: int32[2, 1024]):
    @spmw.work(grid=[2, 4])
    def core():
        x, y = spmw.axes() # axes of the work grid
        local_A = A.shard([None, y]) # replicate first dimension, shard second dimension along axis y
        local_B: int32[1, 256] = B.shard([x, y]) # shard both dimensions
        wid: index = x.id # coordinate along axis x
        local_B[:] = local_A + wid
```


## Programs
An Allo program consists of [**functions**](#function-definition) annotated with specific decorators. Currently supported decorators include:

* [`@kernel`](#kernels): Defines a basic Allo kernel.
* 🔀[`@unit()`](#units): Defines an SPMW module.
* 🔀[`@work(grid: list[int])`](#works): Defines a work within an SPMW module. 
    * A function annotated with `@work` must be declared inside a `@unit` function and cannot exist independently.

### Kernels
<!-- TODO -->

### 🔀Units
A Unit is a function decorated with `@unit()`, which declares a dataflow module. The function signature **must not have a return value**. The parameter list specifies the module's inputs and outputs: inputs are read-only, outputs are write-only, and they **must not overlap**.

A unit defines a namespace for the dataflow module. Inside the function body, you can define resources (e.g., [`stream`](#stream)) and [`work`](#works) that belong to this module. 
**Units cannot be nested.**

For example:
```python
@spmw.unit()
def top(A: int32[16, 16], B: int32[16, 16]):
    # define stream arrays
    ...

    # work1
    ...
    # work2
    ...
```

**Restrictions**: forward reference is currently not supported. Please declare resources before works.

#### Symbols in a Unit
Symbols include:
* The parameters of the unit
* Names of resource declared inside the unit (e.g., streams).

### 🔀Works
A function decorated with `@work(grid: list[int])` declares a set of work instances defined by the same work program. The function has no arguments and **must not have a return value**. 
The decorator parameters are:
* `grid`: Defines a high-dimensional grid, where each grid point corresponds to one work instance. Must be a list of [compile-time constants](#compile-time-constants) (positive integer). Grid axes are 0-based integers.
    * For example:
        ```txt
        Kernel mapping: [P0, P1, P2] -> 3D grid with axes 0, 1, 2
                         |   |   |
                         0   1   2  <- grid axes
        ```

The function body describes the behavior of this work instance.
Call expressions inside the function body can invoke **other units**, enabling hierarchical design.

For example:
```python
@spmw.unit()
def top(A: int32[1024], B: int32[1024]):
    # grid: (1,)
    @spmw.work(grid=[1])
    def core():
        B[:] = A + 1
```

```python
@spmw.unit()
def top(A: int32[1024], B: int32[1024]):
    # grid: (4,)
    @spmw.work(grid=[4])
    def core():
        x = spmw.axes() # axes of the work grid
        local_A = A.shard([x])
        # type annotation is optional
        local_B: int32[256] = B.shard([x])
        local_B[:] = local_A + 1
```
**Explanation**: This defines works over a 1D grid of size 4.
`spmw.axes()` returns the grid axes. Since the grid is 1D, `x` is the only axis.
`A.shard([x])` specifies that the first dimension of the 1D tensor is sharded along grid dimension `x`.
Since the global tensor size is 1024 and the grid size is 4, each work instance receives a local tile of size: `1024 // 4 = 256`.
For each grid point:
* `local_A` refers to a 256-element shard of `A`
* `local_B` refers to a 256-element shard of `B`

Respectively, the works operate on disjoint slices:
* `A[0:256]` -> Work 0 -> `B[0:256]`
* `A[256:512]` -> Work 1 -> `B[256:512]`
* `A[512:768]` -> Work 2 -> `B[512:768]`
* `A[768:1024]` -> Work 3 -> `B[768:1024]`

#### Scoping Rules
All [symbols](#symbols-in-a-unit) defined inside the top unit are visible to its works. 

## Others
### Templates
<!-- TODO -->

### Compile-Time Constants
<!-- TODO: different levels -->
