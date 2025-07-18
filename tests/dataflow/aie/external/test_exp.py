import os
import torch
import torch.nn.functional as F
from typing import Annotated
from allo.ir.types import float32, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.experimental.external_kernel import ExternalModule

VEC_LEN = 64
Ly = Layout("S1S0")

def _test_masked_softmax_tiled():
    exp_kernel = ExternalModule(
        top="my_exp",
        impl_path= "exp.cc",
        input_idx=[0],
        output_idx=[1],
    )

    Ty = float32
    Ty_1 = int32

    @df.region()
    def top():
        @df.kernel(mapping=[1, 1])
        def core(
            Input: Ty[1, VEC_LEN] @ Ly,
            Output: Ty[1, VEC_LEN] @ Ly,
        ):
            exp_kernel(Input, Output)

    # Create random input data
    input_tensor = np.random.randint(-16, 16, (1, VEC_LEN)).astype(np.float32)
    output = np.exp2(input_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(
            top,
            target="aie-mlir",
            profile=True,
            warmup=20,
            num_iters=100,  # ! executing only once may get undefined result.
        )
        output_allo = np.zeros((1, VEC_LEN)).astype(np.float32)
        mod(input_tensor, output_allo)
        np.testing.assert_allclose(output_allo, output, rtol=1e-2)
        print("PASS!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    _test_masked_softmax_tiled()
