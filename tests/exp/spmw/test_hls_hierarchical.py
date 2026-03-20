# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import tempfile
from allo.exp import build, to_hls
import allo
from allo.ir.types import int32, float32, ConstExpr
from allo import spmw
from allo.backend import hls


def test_basic():
    @spmw.unit()
    def vadd(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[1])
        def core():
            for i in allo.grid(1024):
                B[i] = A[i] + 1

    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[1])
        def core():
            vadd(A, B)

    build(top)
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            A = np.random.randint(0, 100, (1024,), dtype=np.int32)
            B = np.random.randint(0, 100, (1024,), dtype=np.int32)
            mod = to_hls(top, project=tmpdir)
            mod(A, B)
            np.testing.assert_allclose(A + 1, B)


def test_multi_instance_v1():
    @spmw.unit()
    def vadd(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[1])
        def core():
            for i in allo.grid(1024):
                B[i] = A[i] + 1

    @spmw.unit()
    def top(A0: int32[1024], A1: int32[1024], B: int32[1024], C: int32[1024]):
        @spmw.work(grid=[1])
        def core():
            vadd(A0, B)
            vadd(A1, C)

    build(top)
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            A = np.random.randint(0, 100, (1024,), dtype=np.int32)
            B = np.zeros((1024,), dtype=np.int32)
            C = np.zeros((1024,), dtype=np.int32)
            mod = to_hls(top, project=tmpdir)
            mod(A, A, B, C)
            # assert fail due to backend issues
            np.testing.assert_allclose(A + 1, C)
            np.testing.assert_allclose(A + 1, B)

def test_multi_instance_v2():
    M, N, K = 32, 32, 32
    P0, P1 = 2, 2

    @spmw.unit()
    def inner(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        @spmw.work(grid=[P0, P1])
        def gemm():
            x, y = spmw.axes()
            pi, pj = x.id, y.id
            Mt: ConstExpr[int32] = M // P0
            Nt: ConstExpr[int32] = N // P1
            for i in range(pi * Mt, (pi + 1) * Mt):
                for j in range(pj * Nt, (pj + 1) * Nt):
                    for k in range(K):
                        C[i, j] += A[i, k] * B[k, j]

    @spmw.unit()
    def top(A: float32[M, K], B: float32[K, N], C1: float32[M, N], C2: float32[M, N]):
        @spmw.work(grid=[2])
        def wrapper():
            x = spmw.axes()
            if x.id == 0:
                inner(A, B, C1)
            else:
                inner(A, B, C2)

    build(top)
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            A = np.random.rand(M, K).astype(np.float32)
            B = np.random.rand(K, N).astype(np.float32)
            C1 = np.zeros((M, N), dtype=np.float32)
            C2 = np.zeros((M, N), dtype=np.float32)
            mod = to_hls(top, project=tmpdir)
            mod(A, B, C1, C2)
            np.testing.assert_allclose(C1, np.dot(A, B), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(C2, np.dot(A, B), rtol=1e-5, atol=1e-5)
            print("Success!")

def test_template_v1():
    @spmw.unit()
    def vadd[L](A: int32[L], B: int32[L]):
        @spmw.work(grid=[1])
        def core():
            for i in allo.grid(L):
                B[i] = A[i] + 1

    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[1])
        def core():
            vadd[1024](A, B)

    build(top)
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            A = np.random.randint(0, 100, (1024,), dtype=np.int32)
            B = np.random.randint(0, 100, (1024,), dtype=np.int32)
            mod = to_hls(top, project=tmpdir)
            mod(A, B)
            np.testing.assert_allclose(A + 1, B)


def test_template_v2():
    M, N, K = 32, 32, 32

    @spmw.unit()
    def inner[P0, P1](A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        @spmw.work(grid=[P0, P1])
        def gemm():
            x, y = spmw.axes()
            pi, pj = x.id, y.id
            Mt: ConstExpr[int32] = M // P0
            Nt: ConstExpr[int32] = N // P1
            for i in range(pi * Mt, (pi + 1) * Mt):
                for j in range(pj * Nt, (pj + 1) * Nt):
                    for k in range(K):
                        C[i, j] += A[i, k] * B[k, j]

    @spmw.unit()
    def top1(A: float32[M, K], B: float32[K, N], C1: float32[M, N], C2: float32[M, N]):
        @spmw.work(grid=[2])
        def wrapper():
            x = spmw.axes()
            if x.id == 0:
                inner[2, 2](A, B, C1)
            else:
                inner[4, 4](A, B, C2)

    build(top1)
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            A = np.random.rand(M, K).astype(np.float32)
            B = np.random.rand(K, N).astype(np.float32)
            C1 = np.zeros((M, N), dtype=np.float32)
            C2 = np.zeros((M, N), dtype=np.float32)
            mod = to_hls(top1, project=tmpdir)
            mod(A, B, C1, C2)
            np.testing.assert_allclose(C1, np.dot(A, B), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(C2, np.dot(A, B), rtol=1e-5, atol=1e-5)
            print("Success!")

    @spmw.unit()
    def top2(A: float32[M, K], B: float32[K, N], C1: float32[M, N], C2: float32[M, N]):
        @spmw.work(grid=[2])
        def wrapper():
            x = spmw.axes()
            if x.id == 0:
                inner[2, 2](A, B, C1)
            else:
                inner[2, 2](A, B, C2)

    build(top2)
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            A = np.random.rand(M, K).astype(np.float32)
            B = np.random.rand(K, N).astype(np.float32)
            C1 = np.zeros((M, N), dtype=np.float32)
            C2 = np.zeros((M, N), dtype=np.float32)
            mod = to_hls(top2, project=tmpdir)
            mod(A, B, C1, C2)
            np.testing.assert_allclose(C1, np.dot(A, B), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(C2, np.dot(A, B), rtol=1e-5, atol=1e-5)
            print("Success!")


if __name__ == "__main__":
    test_basic()
    test_multi_instance_v1()
    test_multi_instance_v2()
    test_template_v1()
    test_template_v2()
