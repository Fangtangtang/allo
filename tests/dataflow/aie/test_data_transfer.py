# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32, float32
import allo.dataflow as df
from allo.backend.aie import call_mlir
import numpy as np
from allo.memory import Layout

Ty = int16
M, N = 64, 64
Pm, Pn = 1, 1
Mt, Nt = M // Pm, N // Pn

Ly = Layout("S1S0")

@df.region()
def top():
    @df.kernel(mapping=[Pm, Pn])
    def transfer(A: Ty[M, N] @ Ly, B: Ty[M, N] @ Ly):
        B[:,:] = A

def test_cooperative_transfer():
    mod = df.build(top, target="aie",profile=True)
    A = np.random.randint(0, 64, (M, N)).astype(np.int16)
    B = np.random.randint(0, 64, (M, N)).astype(np.int16)
    mod(A, B)
    np.testing.assert_allclose(B, A, atol=1e-5)
    print("PASSED!")

def test_cooperative_transfer_mlir(projrct_dir:str):
    A = np.random.randint(0, 64, (M, N)).astype(np.int16)
    B = np.random.randint(0, 64, (M, N)).astype(np.int16)
    call_mlir(
       projrct_dir, Ty, 4096, A, B
    )
    np.testing.assert_allclose(B, A, atol=1e-5)
    print("PASSED!")

if __name__ == "__main__":
    test_cooperative_transfer()
    # test_cooperative_transfer_mlir("top.prj")