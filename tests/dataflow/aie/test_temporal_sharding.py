# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import re

import allo
from allo.ir.types import int16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout, Axis
from allo.backend.aie import is_available

S = Layout.Shard
R = Layout.Replicate


def test_gemm():
    TyI, TyO = int16, int16
    M, N, K = 256, 256, 1024
    Pm, Pn, Pk = 4, 4, 1024 // 64

    Sm = Axis.Spatial(Pm, name="m")
    Sn = Axis.Spatial(Pn, name="n")
    Tk = Axis.Temporal(Pk, name="k")

    LyA = [S(0), S(2)]
    LyB = [S(2), S(1)]
    LyC = [S(0), S(1)] # not shard on temporal dim -> only access once

    @df.region()
    def top(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
        @df.kernel(mapping=[Sm, Sn, Tk], args=[A, B, C])
        def gemm(
            A_port: TyI[M, K] @ LyA,
            B_port: TyI[K, N] @ LyB,
            C_port: TyO[M, N] @ LyC,
        ):
            acc: TyO[M // Pm, N // Pn] = 0

            for _ in range(Pk):
                acc[:, :] = allo.add(
                    allo.matmul(A_port.get(), B_port.get()),
                    acc,
                )

            C_port.put(acc)

    if is_available():
        mod = df.build(top, target="aie")
        A = np.random.randint(-8, 8, (M, K)).astype(np.int16)
        B = np.random.randint(-8, 8, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_mapping_gemm():
    TyI, TyO = int16, int16
    M, N, K = 1024, 128, 1024
    Pm, Pn, Pk = M // 64, N // 64, K // 64

    Sm = Axis.Spatial(Pm, name="m")
    Sn = Axis.Spatial(Pn, name="n")
    Tk = Axis.Temporal(Pk, name="k")

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
            acc: TyO[M // Pm, N // Pn] = 0

            for _ in range(Pk):
                acc[:, :] = allo.add(
                    allo.matmul(A_port.get(), B_port.get()),
                    acc,
                )

            C_port.put(acc)

    if is_available():
        row_num, col_num = 4, 1
        mapping_primitives = []
        for i in range(row_num):
            for j in range(col_num):
                bundle_list = []
                for p in range(Pm // row_num):
                    for q in range(Pn // col_num):
                        bundle_list.append(f"gemm_{i + row_num * p}_{j + col_num * q}")
                if len(bundle_list) > 1:
                    mapping_primitives.append(("bundle", bundle_list))
        print(len(mapping_primitives))
        mod = df.build(top, target="aie", mapping_primitives=mapping_primitives)
        A = np.random.randint(-8, 8, (M, K)).astype(np.int16)
        B = np.random.randint(-8, 8, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_gemm()
    test_mapping_gemm()
