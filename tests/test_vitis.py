# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.ir.types import uint64, uint256, int32, float32, int16, bool
from allo.utils import get_np_struct_type
import allo.dataflow as df
from allo.backend import hls


# ##############################################################
# Test basic
# ##############################################################
def test_grid_for_gemm():
    # from `test_builder.py`, with return value

    # This test is to make sure the whole flow works properly.
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        # Use grid_for with name annotation
        for i, j, k in allo.grid(32, 32, 32, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    # 1. Create customization
    s = allo.customize(gemm)
    print(s.module)

    # 2. Apply transformations and make sure each step the module can be printed
    s.split("i", 8)
    print(s.module)
    s.split("j", 8)
    print(s.module)
    s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
    print(s.module)
    # Make sure the generated loops are correct and ordered
    loops = s.get_loops()
    expected = ["i.outer", "j.outer", "i.inner", "j.inner", "k"]
    assert expected == list(loops.C.loops.keys())

    # 5. HLS CSIM
    if hls.is_available("vitis_hls"):
        np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_C = np.matmul(np_A, np_B)
        hls_mod = s.build(
            target="vitis_hls",
            mode="sw_emu",
            project=f"test_grid_for_gemm.prj",
        )
        sw_out = np.zeros((32, 32), dtype=np.int32)
        hls_mod(np_A, np_B, sw_out)
        np.testing.assert_allclose(sw_out, np_C, atol=1e-3)
        print("Passed HLS test!")


def test_vitis_gemm_template_int32():
    # from `test_vhls.py`
    def gemm[T, M, N, K](A: "T[M, K]", B: "T[K, N]") -> "T[M, N]":
        C: T[M, N] = 0
        for i, j, k in allo.grid(M, N, K, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm, instantiate=[int32, 32, 32, 32])
    if hls.is_available("vitis_hls"):
        mod = s.build(
            target="vitis_hls", mode="sw_emu", project=f"gemm_vitis_{int32}.prj"
        )
        np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_C = np.matmul(np_A, np_B)
        np_C_allo = np.zeros((32, 32), dtype=np.int32)
        mod(np_A, np_B, np_C_allo)
        np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-4)
        print("Passed!")


def test_vitis_gemm_template_float32():
    # from `test_vhls.py`
    def gemm[T, M, N, K](A: "T[M, K]", B: "T[K, N]") -> "T[M, N]":
        C: T[M, N] = 0
        for i, j, k in allo.grid(M, N, K, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm, instantiate=[float32, 64, 64, 64])
    if hls.is_available("vitis_hls"):
        mod = s.build(
            target="vitis_hls", mode="sw_emu", project=f"gemm_vitis_{float32}.prj"
        )
        np_A = np.random.random(size=(64, 64)).astype(np.float32)
        np_B = np.random.random(size=(64, 64)).astype(np.float32)
        np_C = np.matmul(np_A, np_B)
        np_C_allo = np.zeros((64, 64), dtype=np.float32)
        mod(np_A, np_B, np_C_allo)
        np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-4)
        print("Passed!")


def test_vitis_io_stream():
    # from `test_vhls.py`
    def foo(A: int32[32, 32], B: int32[32, 32]):
        pass

    def top(A: int32[32, 32]) -> int32[32, 32]:
        B: int32[32, 32]
        foo(A, B)
        return B

    s = allo.customize(top)
    s.dataflow("top")
    if hls.is_available("vitis_hls"):
        hls_mod = s.build(target="vitis_hls", mode="sw_emu", project="test_io.prj")
        print(s.module)
        np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_B = np.zeros((32, 32), dtype=np.int32)
        hls_mod(np_A, np_B)


# ##############################################################
# Test large bitwidth
# ##############################################################
def test_vadd_adv():
    VLEN = 256
    ELEN = 32

    np_256 = get_np_struct_type(VLEN)

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def VEC(
            A: uint256[1],
            B: uint256[1],
            C: uint256[1],
        ):
            for i in allo.grid(VLEN // ELEN, name="vec_nest"):
                C[0][i * ELEN : (i + 1) * ELEN] = (
                    A[0][i * ELEN : (i + 1) * ELEN] + B[0][i * ELEN : (i + 1) * ELEN]
                )

    A = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    B = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    C = np.zeros(
        VLEN // ELEN,
    ).astype(np.uint32)
    packed_A = np.ascontiguousarray(A).view(np_256)
    packed_B = np.ascontiguousarray(B).view(np_256)
    packed_C = np.ascontiguousarray(C).view(np_256)

    mod = df.build(top, target="simulator")
    mod(packed_A, packed_B, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(A + B, unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED!")

    s = df.customize(top)
    # unroll the lanes
    nest_loop_i = s.get_loops("VEC_0")["vec_nest"]["i"]
    s.unroll(nest_loop_i)
    print(s.module)

    if hls.is_available("vitis_hls"):
        print("Starting Test...")
        mod = s.build(
            target="vitis_hls",
            mode="sw_emu",
            project=f"vec_adv_sw_emu.prj",
            wrap_io=False,
        )
        mod(packed_A, packed_B, packed_C)
        unpacked_C = packed_C.view(np.uint32)
        np.testing.assert_allclose(A + B, unpacked_C, rtol=1e-5, atol=1e-5)
        print(unpacked_C)
        print("Passed Test!")


if __name__ == "__main__":
    test_vadd()
    test_vadd_adv()
    test_grid_for_gemm()
    test_vitis_gemm_template_int32()
    test_vitis_gemm_template_float32()
    test_vitis_io_stream()
