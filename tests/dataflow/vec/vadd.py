# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.ir.types import int256, uint32, uint8, bool, int8, int16, int32, index
import allo.dataflow as df
from allo.utils import get_np_struct_type
from allo.backend import hls

VLEN = 256
ELEN = 32
np_256 = get_np_struct_type(VLEN)


def test_vadd():
    INST_NUM = 4

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def VEC(
            A: int256[INST_NUM],
            B: int256[INST_NUM],
            C: int256[INST_NUM],
        ):
            A_: int256[INST_NUM] = A
            B_: int256[INST_NUM] = B
            C_: int256[INST_NUM]
            for outer in allo.grid(INST_NUM, name="inst_outer"):
                left: index = 0
                for _ in allo.grid(VLEN // ELEN, name="vec_nest"):
                    C_[outer][left : left + ELEN] = (
                        A_[outer][left : left + ELEN] + B_[outer][left : left + ELEN]
                    )
                    left = left + ELEN

            for outer in allo.grid(INST_NUM, name="inst_outer_"):
                C[outer] = C_[outer]

    A = np.random.randint(0, 64, (INST_NUM, VLEN // ELEN)).astype(np.uint32)
    B = np.random.randint(0, 64, (INST_NUM, VLEN // ELEN)).astype(np.uint32)
    C = np.zeros((INST_NUM, VLEN // ELEN)).astype(np.uint32)
    packed_A = np.ascontiguousarray(A).view(np_256).reshape(INST_NUM)
    packed_B = np.ascontiguousarray(B).view(np_256).reshape(INST_NUM)
    packed_C = np.ascontiguousarray(C).view(np_256).reshape(INST_NUM)

    s = df.customize(top)
    s.pipeline(s.get_loops("VEC_0")["inst_outer"]["outer"])
    print(s.module)

    if hls.is_available("vitis_hls"):
        print("Starting Test...")
        mod = s.build(
            target="vitis_hls",
            mode="hw_emu",
            project=f"vadd.prj",
            wrap_io=False,
        )
        mod(packed_A, packed_B, packed_C)
        unpacked_C = packed_C.view(np.uint32).reshape(INST_NUM, VLEN // ELEN)
        np.testing.assert_allclose(A + B, unpacked_C, rtol=1e-5, atol=1e-5)
        print(unpacked_C)
        print("Passed Test!")


if __name__ == "__main__":
    test_vadd()
