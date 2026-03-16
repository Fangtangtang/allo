<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->
# Allo Builtin Library Extensions

This directory serves as an extensible library for handling builtin functions and operations within the Allo compiler.

## Extensibility

The builtin system is designed to be easily extensible. New builtin functions can be added by implementing a handler class that inherits from `BuiltinHandler` and decorating it with `@register_builtin_handler`. This allows the compiler to support new operations without modifying the core `IRBuilder`.

For custom user-defined handlers, the `@register_custom_handler` decorator provides a lightweight alternative that automatically creates a stub Python function linked to the handler.

## Files

- **`handler.py`**:
  - Defines the abstract base class `BuiltinHandler` with three overridable methods:
    - `build()` — constructs MLIR operations for the function call.
    - `infer()` (static) — infers result and operand types.
    - `get_affine_expr()` — returns an affine expression for affine-compatible operations.
  - Implements the registration mechanism via `register_builtin_handler` and `register_custom_handler` decorators.
  - Maintains the `BUILTIN_HANDLERS` registry mapping function names to their respective handler classes.

- **`arith.py`**:
  - Implements handlers for standard arithmetic and comparison operations:
    - **Arithmetic**: `Add`, `Sub`, `Mult`, `Div`, `FloorDiv`, `Mod`.
    - **Comparison**: `Eq`, `NotEq`, `Lt`, `LtE`, `Gt`, `GtE`.
  - Dispatches operations to appropriate MLIR dialects (`arith`, `allo`, `linalg`) based on input types (scalar vs. tensor, integer vs. float vs. fixed-point).
  - Handles signed/unsigned attributes and broadcasting requirements.

- **`value.py`**:
  - Provides handlers for value-related operations:
    - `ConstantHandler` / `ConstantTensorHandler` — scalar and tensor constant creation.
    - `CastHandler` and specialized cast handlers (`IndexCast`, `SIToFP`, `UIToFP`, `FPToSI`, `FPToUI`, `FloatToFixed`, `FixedToFloat`, etc.) — comprehensive type conversion coverage across integer, float, fixed-point, and index types.
    - `BroadcastHandler` — broadcasts a scalar or lower-rank tensor to a target shape via `linalg.broadcast`.

- **`construct.py`**:
  - Implements handlers for hardware construct operations:
    - **Stream operations**: `StreamHandler` (create global stream), `StreamPutHandler` (write to stream), `StreamGetHandler` (read from stream) — with affine-map-based indexed access.
    - **Bit operations**: `SetBitsHandler` (set bit/slice), `GetBitsHandler` (get bit/slice) — supports both single-bit and slice-based manipulation.

- **`meta.py`**:
  - Implements the `WidHandler` for the `get_wid` meta-programming intrinsic.
  - Used in SPMW (Single Program Multiple Workers) contexts: inserts a function declaration for the grid's `get_wid` and emits a call to retrieve worker IDs.
