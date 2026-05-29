# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from allo.exp import process
from allo.ir.types import int32, UInt
from allo.spmw import kernel


def test_get_bit():
    @kernel
    def kernel1(a: int32) -> int32:
        return a[28:32]

    s = process(kernel1)
    pass

    @kernel
    def kernel2(a: int32) -> int32:
        return a[28 : int32.bits]

    s = process(kernel2)

    @kernel
    def kernel3() -> int32:
        a: int32
        a[28 : int32.bits] = 1
        return a

    s = process(kernel3)

    @kernel
    def kernel4(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = (A[i] + 1)[0]
        return B

    s = process(kernel4)
    np_A = np.random.randint(10, size=(10,))
    np.testing.assert_allclose(s(np_A), (np_A + 1) & 1, rtol=1e-5, atol=1e-5)


def test_get_bit_slice():
    @kernel
    def kernel1(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = (A[i] + 1)[0:2]
        return B

    s = process(kernel1)
    np_A = np.random.randint(10, size=(10,))
    np.testing.assert_allclose(s(np_A), (np_A + 1) & 0b11, rtol=1e-5, atol=1e-5)


def test_set_bit_tensor():
    @kernel
    def kernel1(A: UInt(1)[10], B: int32[10]):
        for i in range(10):
            b: int32 = B[i]
            b[0] = A[i]
            B[i] = b

    s = process(kernel1)
    np_A = np.random.randint(2, size=(10,))
    np_B = np.random.randint(10, size=(10,))
    golden = np_B & 0b1110 | np_A
    s(np_A, np_B)
    assert np.array_equal(golden, np_B)

    @kernel
    def kernel2(A: UInt(1)[10], B: int32[10]):
        for i in range(10):
            B[i][0] = A[i]

    s = process(kernel2)
    np_A = np.random.randint(2, size=(10,))
    np_B = np.random.randint(10, size=(10,))
    golden = np_B & 0b1110 | np_A
    s(np_A, np_B)
    assert np.array_equal(golden, np_B)


def test_set_slice():
    @kernel
    def kernel1(A: UInt(2)[10], B: int32[10]):
        for i in range(10):
            B[i][0:2] = A[i]

    s = process(kernel1)
    np_A = np.random.randint(2, size=(10,))
    np_B = np.random.randint(7, size=(10,))
    golden = (np_B & 0b1100) | np_A
    s(np_A, np_B)
    assert np.array_equal(golden, np_B)


def test_dynamic_index():
    @kernel
    def kernel1(A: int32, B: int32[11]):
        for i in range(1, 12):
            B[i - 1] = A[i - 1]

    s = process(kernel1)
    np_B = np.zeros((11,), dtype=np.int32)
    s(1234, np_B)
    print("".join([str(np_B[i]) for i in range(10, -1, -1)]))
    assert bin(1234) == "0b" + "".join([str(np_B[i]) for i in range(10, -1, -1)])


def test_dynamic_slice():
    @kernel
    def kernel1(A: int32, B: int32[11]):
        for i in range(1, 12):
            B[i - 1] = A[i - 1 : i]

    s = process(kernel1)
    np_B = np.zeros((11,), dtype=np.int32)
    s(1234, np_B)
    assert bin(1234) == "0b" + "".join([str(np_B[i]) for i in range(10, -1, -1)])


if __name__ == "__main__":
    test_get_bit()
    test_get_bit_slice()
    test_set_bit_tensor()
    test_set_slice()
    test_dynamic_index()
    test_dynamic_slice()
