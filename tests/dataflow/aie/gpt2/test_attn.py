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

N = 512
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


ATTN_P0 = N // 32
ATTN_P1 = N // 32
ATTN_SCORE_M_TILE = ATTN_P0 * 32
ATTN_SCORE_N_TILE = ATTN_P1 * 32
ATTN_SCORE_LyA = Layout("S0R")
ATTN_SCORE_LyB = Layout("S1R")
ATTN_SCORE_LyC = Layout("S0S1")


def gen_attn_score_primitives():
    ROW = 4
    COL = 4
    primitives = []
    for row in range(ROW):
        for col in range(COL):
            nodes = []
            for j_ in range(ATTN_P0 // COL):
                nodes.extend(
                    [f"core_{ROW*i_+row}_{COL*j_+col}" for i_ in range(ATTN_P1 // ROW)]
                )
            if len(nodes) > 1:
                primitives.append(("bundle", nodes))
    return primitives


@df.region()
def attn_score_kernel():
    @df.kernel(mapping=[ATTN_P0, ATTN_P1])
    def core(
        A: Ty[N, D] @ ATTN_SCORE_LyA,
        B: Ty[N, D] @ ATTN_SCORE_LyB,
        C: Ty[N, N] @ ATTN_SCORE_LyC,
    ):
        attn_score(A, B, C)


attn_score_mod = df.build(
    attn_score_kernel,
    target="aie-mlir",
    project="attn_score.prj",
    mapping_primitives=gen_attn_score_primitives(),
    profile=True,
    warmup=200,
    num_iters=1000,
)


SOFTMAX_P0 = N // 8
SOFTMAX_Ly = Layout("S0R")


def gen_softmax_primitives():
    SOFTMAX_ROW = 4
    primitives = []
    for row in range(SOFTMAX_ROW):
        if SOFTMAX_P0 // SOFTMAX_ROW > 1:
            primitives.append(
                (
                    "bundle",
                    [
                        f"core_{SOFTMAX_ROW*i_+row}"
                        for i_ in range(SOFTMAX_P0 // SOFTMAX_ROW)
                    ],
                )
            )
    return primitives


@df.region()
def softmax_kernel():
    @df.kernel(mapping=[SOFTMAX_P0])
    def core(
        input_x: Ty[N, N] @ SOFTMAX_Ly,
        output_x: Ty[N, N] @ SOFTMAX_Ly,
    ):
        softmax(input_x, output_x)


softmax_mod = df.build(
    softmax_kernel,
    target="aie-mlir",
    project="softmax.prj",
    mapping_primitives=gen_softmax_primitives(),
    profile=True,
    warmup=200,
    num_iters=1000,
)

Mt, Nt = 64, 64
Pk, Pm, Pn = N // 64, N // 64, D // 64
LyA = Layout("S1S2")
LyB = Layout("S2S0")
LyC = Layout("S1S0")


@df.region()
def top():
    pipe = df.array(df.pipe(dtype=Ty, shape=(Mt, Nt), depth=2), shape=(Pk - 1, Pm, Pn))

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


def gen_gemm_primitive():
    ROW = 4
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

    if Pm // ROW > 1:
        for i in range(ROW):
            bundle_list = []
            for p in range(Pm // ROW):
                bundle_list.append(bases[i + ROW * p][0])
            # print(bundle_list)
            mapping_primitives.append(("bundle", bundle_list))

    return mapping_primitives


gemm_mod = df.build(
    top,
    project="gemm.prj",
    target="aie-mlir",
    mapping_primitives=gen_gemm_primitive(),
    profile=True,
    warmup=200,
    num_iters=1000,
)

attention_score = np.empty((N, N), dtype=np_bfloat16)
attn_score_mod(Q, K, attention_score)
attn_weight = np.zeros((N, N)).astype(np_bfloat16)
softmax_mod(attention_score, attn_weight)
x = np.zeros((N, D)).astype(np_bfloat16)
gemm_mod(attn_weight, V, x)
