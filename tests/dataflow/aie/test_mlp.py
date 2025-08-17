# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import allo
from allo.ir.types import int4, int8
import allo.dataflow as df
from allo.memory import Layout
from ml_dtypes import bfloat16 as np_bfloat16

seq_len = 4
d_model = 768
hidden = 3072

# X = np.random.randn(seq_len, d_model)
# W1 = np.random.randn(d_model, hidden)
# W2 = np.random.randn(hidden, d_model)
# Y2 = np.zeros((seq_len, d_model))

# tile_x = 64
# tile_w1 = 32
# tile_w2 = 32

# for i in range(0, d_model, tile_w2):
#     for j in range(0, hidden, tile_w1):
#         t_w2 = W2[j : j + tile_w1, i : i + tile_w2]
#         for k in range(0, d_model, tile_x):
#             t_x = X[:, k : k + tile_x]
#             t_w1 = W1[k : k + tile_x, j : j + tile_w1]
#             tile1_out = t_x @ t_w1
#             Y2[:, i : i + tile_w2] += tile1_out @ t_w2

# # print(Y2)
# # print((X @ W1) @ W2)
# # np.testing.assert_allclose(Y2, (X @ W1) @ W2, rtol=1e-5)


def _test_gemm_1D():
    Ty = int8
    Ty_l = int4
    M, N, K = 16, 16, 16
    P0 = 1
    
    LyA = Layout("S0R")
    LyB = Layout("RS1")
    LyC = Layout("S0S1")


    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def gemm(A: Ty[M, K] @ LyA, B: Ty_l[K, N], C: Ty[M, N] @ LyA):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(top, target="aie-mlir")
    A = np.random.random((M, K)).astype(np.int8)
    B = np.random.random((K, N)).astype(np.int8)
    C = np.zeros((M, N)).astype(np.int8)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")
_test_gemm_1D()