# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie import is_available

Ly = Layout("S0")


def _test_passthrough():
    Ty = int32
    M = 1024
    P0 = 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly):
            B[:] = A

    A = np.random.randint(0, 100, M).astype(np.int32)
    if is_available():
        mod = df.build(top, target="air")
        B = np.zeros(M).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def _test_vector_scalar_add():
    Ty = int32
    M = 1024
    P0 = 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly):
            B[:] = allo.add(A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    if is_available():
        mod = df.build(top, target="air")
        B = np.zeros(M).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def _test_vector_vector_add():
    Ty = float32
    M = 1024
    P0 = 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly, C: Ty[M] @ Ly):
            C[:] = allo.add(A, B)

    A = np.random.random(M).astype(np.float32)
    B = np.random.random(M).astype(np.float32)
    if is_available():
        mod = df.build(top, target="air")
        C = np.zeros(M).astype(np.float32)
        mod(A, B, C)
        np.testing.assert_allclose(C, A + B, rtol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def _test_vector_scalar_add_v2():
    Ty = int32
    M = 1024
    P0 = 4
    in_M = M // P0

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[in_M], B: Ty[M] @ Ly):
            B[:] = allo.add(A, 1)

    A = np.random.randint(0, 100, in_M).astype(np.int32)
    if is_available():
        mod = df.build(top, target="air")
        B = np.zeros(M).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B[0:in_M], A + 1)
        np.testing.assert_allclose(B[in_M : 2 * in_M], A + 1)
        np.testing.assert_allclose(B[2 * in_M : 3 * in_M], A + 1)
        np.testing.assert_allclose(B[3 * in_M : 4 * in_M], A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def _test_vector_scalar_mul():
    Ty = float32
    M = 512
    P0 = 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly):
            B[:] = allo.mul(A, 2)

    A = np.random.random(M).astype(np.float32)
    if is_available():
        mod = df.build(top, target="air")
        B = np.zeros(M).astype(np.float32)
        mod(A, B)
        np.testing.assert_allclose(B, A * 2, rtol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def _test_vector_vector_mul():
    Ty = float32
    M = 1024
    P0 = 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly, C: Ty[M] @ Ly):
            C[:] = allo.mul(A, B)

    A = np.random.random(M).astype(np.float32)
    B = np.random.random(M).astype(np.float32)

    if is_available():
        mod = df.build(top, target="air")
        C = np.zeros(M).astype(np.float32)
        mod(A, B, C)
        np.testing.assert_allclose(C, A * B, rtol=1e-4)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def _test_vector_vector_mul_v2():
    Ty = float32
    M = 1024
    P0 = 4
    in_M = M // P0

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[in_M], B: Ty[in_M], C: Ty[M] @ Ly):
            C[:] = allo.mul(A, B)

    A = np.random.random(in_M).astype(np.float32)
    B = np.random.random(in_M).astype(np.float32)

    if is_available():
        mod = df.build(top, target="air")
        C = np.zeros(M).astype(np.float32)
        mod(A, B, C)
        np.testing.assert_allclose(C[0:in_M], A * B, rtol=1e-4)
        np.testing.assert_allclose(C[in_M : 2 * in_M], A * B, rtol=1e-4)
        np.testing.assert_allclose(C[2 * in_M : 3 * in_M], A * B, rtol=1e-4)
        np.testing.assert_allclose(C[3 * in_M : 4 * in_M], A * B, rtol=1e-4)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    _test_passthrough()
    _test_vector_vector_add()
    _test_vector_scalar_add()
    _test_vector_scalar_add_v2()
    _test_vector_scalar_mul()
    _test_vector_vector_mul()
    _test_vector_vector_mul_v2()
