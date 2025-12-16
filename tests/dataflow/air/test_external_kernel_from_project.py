# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from allo.backend.aie import is_available
from allo.backend import air


def _test_external_kernel():
    M = 4096

    A = np.random.random(M).astype(np.uint8)
    if is_available():
        B = np.zeros(M).astype(np.uint8)
        air._call_prj("external_prj", [1], A, B)
        np.testing.assert_allclose(A, B, rtol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    _test_external_kernel()
