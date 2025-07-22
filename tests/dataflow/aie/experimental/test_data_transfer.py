# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32, float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

Ty = int16
# M, N, K = 16, 16, 32
# Pm, Pn, Pk = 1, 1, 4
M, N, K = 32, 32, 32
Pm, Pn, Pk = 2, 2, 2
Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

LyA = Layout("RR")

@df.region()
def top():
    @df.kernel(mapping=[2, 1])
    def producer(A: Ty[M, N]@LyA,B: Ty[M, N]):
        
        allo.add(A , 1)
        pk, pm = df.get_pid()
        with allo.meta_if(pk == 0):
            B[:, :] = A

    # pipe = df.pipe(dtype=Ty, shape=(M, N), depth=2)

    # @df.kernel(mapping=[2])
    # def producer(A: Ty[M, N]):
    #     pipe.put(A)
        
    # @df.kernel(mapping=[1])
    # def consumer(B: Ty[M, N]):
    #     B[:, :] = pipe.get()

def test_cooperative_gemm():
    mod = df.build(top, target="aie-mlir",project="transfer.prj", profile=True, use_default_codegen=True, trace=[("producer", (0,0,)), ("producer", (1,0,))])
    # mod = df.build(top, target="aie-mlir",project="transfer.prj", use_default_codegen=True)
    A = np.random.randint(0, 64, (M, N)).astype(np.int16)
    B = np.zeros((M, N)).astype(np.int16)
    mod(A, B)
    # np.testing.assert_allclose(B, A, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    test_cooperative_gemm()
