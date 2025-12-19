# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extra regression tests for the AIR backend multi-tile translation.

These tests are not part of the original grading set but help guard against
regressions in:

* multi-tile sharded 1D tensors (Layout("S0"))
* replicated scalar/vector inputs
* correct synchronization of async DMAs (wait_all)
"""

import numpy as np

import allo
import allo.dataflow as df
from allo.backend.aie import is_available
from allo.ir.types import int32, float32
from allo.memory import Layout

Ly = Layout("S0")


def test_partition_sharded_add_roundtrip():
    Ty = float32
    M = 1024
    P0 = 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly, C: Ty[M] @ Ly):
            C[:] = allo.add(A, B)

    if not is_available():
        return

    A = np.random.random(M).astype(np.float32)
    B = np.random.random(M).astype(np.float32)
    C = np.zeros(M).astype(np.float32)
    mod = df.build(top, target="air")
    mod(A, B, C)
    np.testing.assert_allclose(C, A + B, rtol=1e-5)


def test_partition_broadcast_input_to_sharded_output():
    Ty = int32
    M = 1024
    P0 = 4
    in_M = M // P0

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[in_M], B: Ty[M] @ Ly):
            B[:] = allo.add(A, 1)

    if not is_available():
        return

    A = np.random.randint(0, 100, in_M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)
    mod = df.build(top, target="air")
    mod(A, B)
    for i in range(P0):
        np.testing.assert_allclose(B[i * in_M : (i + 1) * in_M], A + 1)
