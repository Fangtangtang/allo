# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from allo.backend.aie import is_available
from allo.backend import air


def _test_passthrough():
    M = 1024

    A = np.random.randint(0, 100, M).astype(np.int32)
    if is_available():
        C = np.zeros(M).astype(np.int32)
        air.convert("passthrough.prj")
        air._call_prj("passthrough.prj", [1], A, C)
        np.testing.assert_allclose(C, A, rtol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def _test_vector_vector_add():
    M = 1024

    A = np.random.random(M).astype(np.float32)
    B = np.random.random(M).astype(np.float32)
    if is_available():
        C = np.zeros(M).astype(np.float32)
        air.convert("vadd.prj")
        air._call_prj("vadd.prj", [2], A, B, C)
        np.testing.assert_allclose(C, A + B, rtol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    _test_passthrough()
    _test_vector_vector_add()
