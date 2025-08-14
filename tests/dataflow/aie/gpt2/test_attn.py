# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import bfloat16
import allo.dataflow as df
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.experimental.external_kernel import ExternalModule

KERNEL_LIB_PATH = "/home/sf668/workspace/allo/allo/backend/experimental/kernels/"
np.random.seed(42)

Ty = bfloat16


def gen_pingpong_gemm_mapping_primitive(Pm, Pn, Pk, col_num=4, row_num=4):
    # chain on k dimension
    mapping_primitives = []
    bases: list[list[str]] = []
    for i in range(Pm):
        bases.append([])
        for j in range(Pn):
            base = f"gemm_0_{i}_{j}"
            for k in range(1, Pk):
                mapping_primitives.append(("chain", [base, f"gemm_{k}_{i}_{j}"]))
                base += f"-gemm_{k}_{i}_{j}"
            bases[i].append(base)

    if Pn // col_num < 1 or Pm // row_num < 1:
        col_num, row_num = row_num, col_num

    if Pn // col_num > 1 or Pm // row_num > 1:
        for i in range(row_num):
            for j in range(col_num):
                bundle_list = []
                for p in range(Pm // row_num):
                    for q in range(Pn // col_num):
                        bundle_list.append(bases[i + row_num * p][j + col_num * q])
                mapping_primitives.append(("bundle", bundle_list))

    return mapping_primitives


def _test_pingpong_gemm(M, N, K, Pm, Pn, Pk, A, B ,C):
    Mt, Nt = M // Pm, N // Pn

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top():
        pipe = df.array(
            df.pipe(dtype=Ty, shape=(Mt, Nt), depth=2), shape=(Pk - 1, Pm, Pn)
        )

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: Ty[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            with allo.meta_if(pk > 0):
                C_in: Ty[Mt, Nt] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in: Ty[Mt, Nt] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

    mapping_primitives = gen_pingpong_gemm_mapping_primitive(
        Pm,
        Pn,
        Pk,
    )

    mod = df.build(
        top,
        project="top.prj",
        target="aie-mlir",
        mapping_primitives=mapping_primitives,
        profile=True,
        warmup=200,
        num_iters=1000,
    )
    mod(A, B, C)

N = 128
D = 64

Q = np.random.randn(N, D).astype(np_bfloat16)
K = np.random.randn(N, D).astype(np_bfloat16)
V = np.random.randn(N, D).astype(np_bfloat16)
O = np.zeros(N * D).astype(np_bfloat16)
attn_score = ExternalModule(
    top="transpose_matmul_with_scale_bfloat16",
    impl_path=KERNEL_LIB_PATH + "transpose_matmul_with_scale.cc",
    input_idx=[0, 1],
    output_idx=[2],
)
softmax = ExternalModule(
    top="softmax_bfloat16",
    impl_path=KERNEL_LIB_PATH + "softmax.cc",
    input_idx=[0],
    output_idx=[1],
)

ATTN_P0 = 4
ATTN_P1 = 4
ATTN_SCORE_M_TILE = ATTN_P0 * 32
ATTN_SCORE_N_TILE = ATTN_P1 * 32
ATTN_SCORE_LyA = Layout("S0R")
ATTN_SCORE_LyB = Layout("S1R")
ATTN_SCORE_LyC = Layout("S0S1")


@df.region()
def attn_score_kernel():
    @df.kernel(mapping=[ATTN_P0, ATTN_P1])
    def core(
        A: Ty[ATTN_SCORE_M_TILE, D] @ ATTN_SCORE_LyA,
        B: Ty[ATTN_SCORE_N_TILE, D] @ ATTN_SCORE_LyB,
        C: Ty[ATTN_SCORE_M_TILE, ATTN_SCORE_N_TILE] @ ATTN_SCORE_LyC,
    ):
        attn_score(A, B, C)


attn_score_mod = df.build(
    attn_score_kernel, target="aie-mlir", project="attn_score.prj"
)

SOFTMAX_P0 = 16
SOFTMAX_TILE = SOFTMAX_P0 * 8
SOFTMAX_Ly = Layout("S0R")


@df.region()
def softmax_kernel():
    @df.kernel(mapping=[SOFTMAX_P0])
    def core(
        input_x: Ty[SOFTMAX_TILE, N] @ SOFTMAX_Ly,
        output_x: Ty[SOFTMAX_TILE, N] @ SOFTMAX_Ly,
    ):
        softmax(input_x, output_x)


softmax_mod = df.build(softmax_kernel, target="aie-mlir", project="softmax.prj")

Pm, Pn, Pk = 2, 2, 2
Mt, Nt = N // Pm, D // Pn

LyA = Layout("S1S2")
LyB = Layout("S2S0")
LyC = Layout("S1S0")

@df.region()
def top():
    pipe = df.array(
        df.pipe(dtype=Ty, shape=(Mt, Nt), depth=2), shape=(Pk - 1, Pm, Pn)
    )

    @df.kernel(mapping=[Pk, Pm, Pn])
    def gemm(A: Ty[N, N] @ LyA, B: Ty[N, D] @ LyB, C: Ty[N, D] @ LyC):
        pk, pm, pn = df.get_pid()
        with allo.meta_if(pk > 0):
            C_in: Ty[Mt, Nt] = pipe[pk - 1, pm, pn].get()
        with allo.meta_else():
            C_in: Ty[Mt, Nt] = 0
        C_out: Ty[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
        with allo.meta_if(pk < Pk - 1):
            pipe[pk, pm, pn].put(C_out)
        with allo.meta_elif(pk == Pk - 1):
            C[:, :] = C_out

gemm_mod = df.build(top, target="aie-mlir", project= "gemm.prj", profile=True)

attention_score = np.empty((N, N), dtype=np_bfloat16)
for i in range(N // ATTN_SCORE_M_TILE):
    for j in range(N // ATTN_SCORE_N_TILE):
        attn_score_mod(
            Q[
                i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE,
                :,
            ],
            K[
                j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE,
                :,
            ],
            attention_score[
                i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE,
                j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE,
            ],
        )

attn_weight = np.zeros((N, N)).astype(np_bfloat16)

for i in range(N // SOFTMAX_TILE):
    softmax_mod(
        attention_score[i * SOFTMAX_TILE : (i + 1) * SOFTMAX_TILE, :],
        attn_weight[i * SOFTMAX_TILE : (i + 1) * SOFTMAX_TILE, :],
    )

x = np.zeros((N, D)).astype(np_bfloat16)
gemm_mod(attn_weight, V, x)
# _test_pingpong_gemm(N, D, N, 2, 2, 2, attn_weight, V ,x)