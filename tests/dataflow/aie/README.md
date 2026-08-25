# Unified Spatial and Temporal Sharding for AIE Dataflow

> **Status: Design proposal — not yet implemented.**
>

## Motivation

An `@df.kernel` annotation should describe how a complete tensor in off-chip
memory is distributed without requiring the programmer to calculate or declare
local tile sizes. The interface needs a clear boundary between global storage
and values that are local to a processing element (PE):

- Kernel tensor annotations always retain the complete global tensor shape.
- Every `@df.kernel` tensor argument is a port handle, not a local tensor.
- Only `get()`, `put(value)`, and inferred allocations such as `zeros()` expose
  compiler-derived local tensors.
- Spatial ownership and temporal transfer order are logical sharding concepts.
  Packing and vector width remain independent backend optimizations.

This separation lets the compiler infer local types, generate DMA offsets, and
check transfer counts while keeping shapes such as `Mt`, `Nt`, and `Kt` out of
the source program.

### Alternatives considered

| Design | Advantages | Limitations |
| --- | --- | --- |
| Separate `df.T(...)` axes and `temporal=[...]` | Makes temporal axes explicit and keeps `mapping` spatial-only. | Duplicates declarations, creates separate axis namespaces, and requires layouts and traversal metadata to refer to the same concept in different places. |
| Treat subtiles as packing size | Closely resembles a physical transfer width and can guide DMA optimization. | Does not express logical ownership, tile count, or multidimensional traversal order; it also couples the public model to a backend optimization. |
| Add an explicit `step` or subtile index | Can express arbitrary access orders. | Exposes mixed-radix scheduling arithmetic to the programmer and makes port synchronization error-prone. |
| Put typed spatial and temporal axes in `mapping` | Provides one source of truth for axis kind, extent, identity, placement, and default traversal. The same `S(i)` syntax composes both kinds of sharding. | Broadens the meaning of `mapping`, makes numeric references sensitive to axis reordering, and requires every compiler stage to distinguish logical axes from physical PE axes. |

The proposed interface adopts the last design. Its additional costs are
intentional: `get_pid()` cannot correspond one-to-one with every entry in
`mapping`, static port-access restrictions require new analysis, and data that
is invariant across a temporal axis must be explicitly saved and reused.

## Proposed public interface

Axes are immutable, typed objects declared directly in the kernel mapping:

```python
Sm = df.Axis.Spatial(Pm, name="m")
Sn = df.Axis.Spatial(Pn, name="n")
Tk = df.Axis.Temporal(Pk, name="k")

@df.kernel(mapping=[Sm, Sn, Tk], args=[A, B, C])
def kernel(...):
    ...
```

The interface follows these rules:

- An axis extent is a positive compile-time integer.
- An axis name is optional and used only for diagnostics. Layouts refer to the
  axis's logical index in `mapping`.
- A legacy integer mapping entry is shorthand for a spatial axis, preserving
  the meaning of existing mappings such as `mapping=[2, 4]`.
- All spatial axes precede all temporal axes in `mapping`. Spatial entries, in
  order, define the physical PE mesh. Temporal entries, in order, define the
  default outer-to-inner traversal; the last temporal axis is fastest.
- `S(i)` shards by logical mapping axis `i`, regardless of whether that axis is
  spatial or temporal.
- Raw string forms such as `S("T")` and a separate `temporal=` kernel argument
  are not part of the final interface.
- `df.get_pid()` continues to return only spatial coordinates. Its tuple length
  is the number of spatial axes, not `len(mapping)`.
- `df.get_axis_id(i)` returns the coordinate of spatial axis `i`, or the current
  coordinate of temporal axis `i` while its explicit `allo.meta_for` is active.
- Ordinary Allo functions and inter-kernel `Stream` objects retain their current
  semantics.

### Ordered shard products

Several axes can shard one tensor dimension through an ordered product:

```python
S(0) * S(2)  # spatial axis 0, then temporal axis 2
S(2) * S(0)  # temporal axis 2, then spatial axis 0
S(0) * S(1)  # two spatial axes shard the same dimension
```

For a global dimension of extent `G` and the ordered product
`X0 * X1 * ... * Xn`, let `Ei` be the extent of `Xi` and `Ci` its coordinate.
The compiler derives:

```text
local_extent = G / product(Ei)
block_id     = (((C0 * E1 + C1) * E2 + C2) ... * En + Cn)
origin       = block_id * local_extent
```

For example, if axes 0 and 2 have extents `P0` and `Pk`:

```text
S(0) * S(2): block_id = pid0 * Pk + tk
S(2) * S(0): block_id = tk * P0 + pid0
```

Both products infer the same leaf extent, but they deliberately select
different global tiles. If a temporal factor precedes a spatial factor, the
tiles owned by one PE can form a non-contiguous set even though every individual
transfer tile remains rectangular.

The compiler requires exact divisibility, valid mapping indices, and at most one
occurrence of any axis in a tensor layout. `Replicate` cannot participate in a
shard product.

## Global, spatial, and local views

The compiler maintains three distinct views of every port tensor:

1. **Global tensor:** the complete shape written in the kernel annotation.
2. **Spatial ownership:** the set of tiles selected after fixing the PE's
   spatial coordinates. This set is not necessarily contiguous for every shard
   product order.
3. **Transfer tile:** the rectangular leaf tensor returned by one `get()` or
   consumed by one `put(value)` after applying every spatial and temporal shard
   factor.

The kernel argument itself always has the first view and acts as a port handle.
The programmer never declares the second or third view as a separate type.

## Port semantics

- `port.get()` returns one inferred transfer tile and advances that input
  port's cursor.
- `port.put(value)` requires `value` to have the inferred transfer-tile type and
  advances that output port's cursor.
- `port.zeros()` creates a zero-initialized transfer-tile value without
  performing a transfer or advancing a cursor.
- Every port has an independent cursor. The first accesses to two different
  ports therefore select temporal coordinate zero on both ports.
- The required transfer count for a port is the product of the extents of the
  temporal axes appearing in that port's layout. A layout with no temporal axis
  implies one transfer.
- Calling `get()` again requests the next tile. Reuse is expressed by saving the
  previously returned local tensor rather than calling `get()` again.

The compiler statically rejects cursor under-consumption and over-consumption,
data-dependent `get()` or `put()` control flow, and a `put()` value whose type
does not match the inferred transfer tile.

### Traversal forms

Temporal traversal has two static source forms.

**Manually unrolled access.** If a port is accessed without an explicit
temporal loop, its cursor follows the temporal axes in `mapping` order,
outermost to innermost. The last applicable temporal axis varies fastest.

**Explicit access.** `allo.meta_for` accepts a temporal axis object. A complete
nest can override the default order, with lexical loop nesting defining the
outer-to-inner traversal:

```python
with allo.meta_for(Touter) as _outer:
    with allo.meta_for(Tinner) as _inner:
        tile = port.get()
```

Every temporal axis participating in that port access must occur exactly once
in the active nest. Partial nests, duplicated axes, and mixing loop-driven and
cursor-driven traversal for one port are invalid. A port may be accessed only
under the temporal axes used by its layout. If a local tile is invariant to an
enclosing temporal axis, it must be fetched outside that loop and explicitly
reused inside it; the compiler does not add implicit caching or duplicate
transfers.

## Example 1: split-K GEMM

The global shapes remain `M x K`, `K x N`, and `M x N`. Spatial axes distribute
the output rows and columns, while one temporal axis streams successive pieces
of the reduction dimension:

```python
Sm = df.Axis.Spatial(Pm, name="m")   # axis 0
Sn = df.Axis.Spatial(Pn, name="n")   # axis 1
Tk = df.Axis.Temporal(Pk, name="k")  # axis 2

LyA = [S(0), S(2)]
LyB = [S(2), S(1)]
LyC = [S(0), S(1)]

@df.region()
def top(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
    @df.kernel(mapping=[Sm, Sn, Tk], args=[A, B, C])
    def gemm(
        A_port: TyI[M, K] @ LyA,
        B_port: TyI[K, N] @ LyB,
        C_port: TyO[M, N] @ LyC,
    ):
        acc = C_port.zeros()

        with allo.meta_for(Tk) as _k:
            acc[:, :] = allo.add(
                allo.matmul(A_port.get(), B_port.get()),
                acc,
            )

        C_port.put(acc)
```

No local dimension names appear in the program. The inferred leaf shapes are:

```text
A_port.get()   -> TyI[M / Pm, K / Pk]
B_port.get()   -> TyI[K / Pk, N / Pn]
C_port.zeros() -> TyO[M / Pm, N / Pn]
C_port.put()   <- TyO[M / Pm, N / Pn]
```

At spatial coordinates `(pm, pn)` and temporal coordinate `k`, the global
origins are:

```text
A: (pm * M / Pm, k  * K / Pk)
B: (k  * K / Pk, pn * N / Pn)
C: (pm * M / Pm, pn * N / Pn)
```

The A and B ports each consume `Pk` tiles. C has no temporal factor, so its local
accumulator is created once and one completed tile is written after the
reduction.

## Example 2: two-dimensional temporal traversal

Consider a global `M x N` tensor spatially divided into four horizontal device
bands. Within every device, temporal axes divide its band into two rows and four
columns of transfer tiles. The desired traversal visits the top and bottom tile
of one column before moving to the next column.

```python
Sd = df.Axis.Spatial(4, name="device")  # axis 0
Tc = df.Axis.Temporal(4, name="col")    # axis 1, outer
Tr = df.Axis.Temporal(2, name="row")    # axis 2, inner

LyX = [S(0) * S(2), S(1)]

@df.region()
def top(X: TyI[M, N], Y: TyO[M, N]):
    @df.kernel(mapping=[Sd, Tc, Tr], args=[X, Y])
    def copy(
        X_port: TyI[M, N] @ LyX,
        Y_port: TyO[M, N] @ LyX,
    ):
        with allo.meta_for(Tc) as _c:
            with allo.meta_for(Tr) as _r:
                Y_port.put(X_port.get())
```

Assuming `M` is divisible by 8 and `N` by 4, the compiler derives:

```text
spatial view per device = [M / 4, N]
transfer tile           = [M / 8, N / 4]

row_origin = (device * 2 + row) * M / 8
col_origin = col * N / 4
```

Because `Tc` is outer and `Tr` is inner, each device's `(row, column)` sequence
is:

```text
(0, 0), (1, 0), (0, 1), (1, 1),
(0, 2), (1, 2), (0, 3), (1, 3)
```

The same default order can be expressed by eight manually unrolled accesses;
the syntax contains no explicit coordinates because each port has its own
cursor:

```python
Y_port.put(X_port.get())  # (row=0, col=0)
Y_port.put(X_port.get())  # (row=1, col=0)
Y_port.put(X_port.get())  # (row=0, col=1)
Y_port.put(X_port.get())  # (row=1, col=1)
Y_port.put(X_port.get())  # (row=0, col=2)
Y_port.put(X_port.get())  # (row=1, col=2)
Y_port.put(X_port.get())  # (row=0, col=3)
Y_port.put(X_port.get())  # (row=1, col=3)
```

Explicit loops may intentionally override traversal without changing placement.
Reversing the nest makes row outer and column inner:

```python
with allo.meta_for(Tr) as _r:
    with allo.meta_for(Tc) as _c:
        Y_port.put(X_port.get())
```

This visits `(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), ...`; the layout and global
origin formulas remain unchanged.

## Future compiler work

Implementation should proceed as one intentional dataflow API change rather
than preserving two meanings for kernel tensor arguments:

1. Add immutable spatial and temporal axis types. Normalize legacy integer
   mapping entries into spatial axes and derive the physical mesh by filtering
   spatial entries from the complete logical mapping.
2. Extend `Shard` with an ordered `ShardProduct`. Normalize layout expressions,
   infer leaf shapes, and represent mixed-radix origins symbolically until both
   spatial and temporal coordinates are known.
3. Preserve global shapes on kernel arguments and introduce an internal tensor
   port type whose `get()`, `put()`, and `zeros()` operations expose inferred
   leaf types.
4. Extend `allo.meta_for` to accept temporal axis objects. Add static traversal,
   cursor-count, access-scope, and invariant-reuse validation.
5. Instantiate functions and AIE cores only over the spatial mesh. Enumerate
   temporal DMA coordinates separately and lower mixed-radix origins into DMA
   or object-FIFO access patterns; temporal axes must not become fake mesh
   dimensions.
6. Migrate existing `@df.kernel` tensor arguments to explicit port operations.
   Keep ordinary functions and inter-kernel streams unchanged.

## Future test plan

Positive coverage should include:

- Split-K GEMM with no user-written local dimensions.
- The four-device, two-dimensional temporal traversal above.
- Identical values and DMA order for explicit loops and manual unrolling.
- Different origins for `S(0) * S(2)` and `S(2) * S(0)`.
- One tensor dimension sharded over two spatial axes.
- An explicit temporal-loop order override.
- Explicit reuse of a port invariant to an enclosing temporal axis.
- Legacy integer mappings and spatial-only `df.get_pid()` behavior.

Static failure tests should cover:

- Non-positive or non-constant axis extents.
- Global extents that are not exactly divisible by their shard products.
- Invalid mapping indices, undeclared axes, or repeated axes in one layout.
- A shard product containing `Replicate`.
- Partial, duplicated, or otherwise invalid explicit temporal-loop nests.
- Data-dependent port accesses or mixed traversal modes.
- Cursor under-consumption or over-consumption.
- A `put()` value with an incompatible leaf shape.

The first implementation remains limited to static rectangular exact-division
sharding. Tails, halos, overlapping windows, arbitrary indexed traversal, and
backend packing policy are deferred.
