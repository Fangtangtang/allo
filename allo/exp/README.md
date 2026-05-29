<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->
# Allo IR Generation

This directory contains the core logic for converting Python AST into Allo Intermediate Representation (IR), which relies on MLIR.


The compilation process is divided into two distinct phases to separate concerns:

1.  **Inference Phase (`ASTPreProcessor`)**:
    - Handles the "dirty work" directly on the AST.
    - Responsible for type inference, semantic checking, and desugaring syntax sugar.
    - Produces a structured, fully-resolved AST where types and shapes are known (and explicitly encoded).

2.  **Builder Phase (`IRBuilder`)**:
    - Consumes the well-formed AST from the Inference Phase.
    - Focuses solely on traversal and IR construction.
    - Does not perform complex analysis; assumes the input AST is correct and fully annotated.

## Components

- **`__init__.py`**:
  - Package entry point exposing the public API: `build()` and `process()`.
  - `build(fn, ...)` orchestrates the full pipeline: creates a `SymbolTable`, runs `ASTPreProcessor`, and invokes `IRBuilder` to produce an MLIR module.
  - `process(fn, ...)` wraps `build()` and returns an `LLVMModule` (allo's cpu backend) for execution.

- **`ast_preprocessor.py`**:
  - Implements the `ASTPreProcessor` class.
  - Transforms the standard Python AST into a more structured version suitable for IR generation.
  - Handles type annotations, broadcasting, casting, and AST simplification.
  - Manages symbol tables and scoping (block-level and function-level) for variables and constants.

- **`ir_builder.py`**:
  - Implements the `IRBuilder` class.
  - Traverses the processed AST to generate MLIR operations.
  - Utilizes various MLIR dialects (e.g., `arith`, `memref`, `scf`, `affine`, `linalg`, `allo`) to construct the actual IR.
  - Manages insertion points and MLIR context.

- **`config.py`**:
  - Defines the `Interface` dataclass for configuring meta-programming, SPMW, and library interfaces path. 
  - Manages the global typing rule configuration (`_TYPING_RULE_CONFIG`), switchable between `"hls"` and other modes.
  - Provides `ir_builder_config_context()`, a context manager for scoped configuration overrides.

- **`utils.py`**:
  - Provides utility classes and functions:
    - `SymbolTable`: global registry for functions, constants, variables, and template instantiations. Provides name mangling for templates and namespaces.
    - `Scope`: lightweight container for block-level constants and variables (used via scope chains in both phases).
    - `ErrorMsg` / `report_error()`: rich-formatted error reporting with source location.
    - `get_ast()`: extracts a deep-copied AST node from function objects.

- **`builtin/`**:
  - A comprehensive, extensible library for handling builtin functions and operations.
  - New builtins are added by subclassing `BuiltinHandler` and decorating with `@register_builtin_handler`.
  - Each handler encapsulates:
    - `build()` — constructs the corresponding MLIR operations.
    - `infer()` (static) — type inference for the operation's result and operand types.
    - `get_affine_expr()` — optional; returns an affine expression if the operation is affine-compatible.
  - See [`builtin/README.md`](builtin/README.md) for detailed per-file documentation.
