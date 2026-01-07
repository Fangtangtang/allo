# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from allo.ir.types import int8, Stream
import allo.dataflow as df
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

S = Layout.Shard
R = Layout.Replicate

rgba2hueLine = ExternalModule(
    top="rgba2hueLine",
    impl_path=f"{os.path.dirname(os.path.abspath(__file__))}/rgba2hue.cc",
    input_idx=[0],
    output_idx=[1],
)


thresholdLine40 = ExternalModule(
    top="thresholdLine40",
    impl_path=f"{os.path.dirname(os.path.abspath(__file__))}/threshold.cc",
    input_idx=[0],
    output_idx=[1],
)

thresholdLine30 = ExternalModule(
    top="thresholdLine30",
    impl_path=f"{os.path.dirname(os.path.abspath(__file__))}/threshold.cc",
    input_idx=[0],
    output_idx=[1],
)

thresholdLine160 = ExternalModule(
    top="thresholdLine160",
    impl_path=f"{os.path.dirname(os.path.abspath(__file__))}/threshold.cc",
    input_idx=[0],
    output_idx=[1],
)

thresholdLine90 = ExternalModule(
    top="thresholdLine90",
    impl_path=f"{os.path.dirname(os.path.abspath(__file__))}/threshold.cc",
    input_idx=[0],
    output_idx=[1],
)

bitwiseORLine = ExternalModule(
    top="bitwiseORLine",
    impl_path=f"{os.path.dirname(os.path.abspath(__file__))}/bitwiseOR.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

gray2rgbaLine = ExternalModule(
    top="gray2rgbaLine",
    impl_path=f"{os.path.dirname(os.path.abspath(__file__))}/gray2rgba.cc",
    input_idx=[0],
    output_idx=[1],
)

bitwiseANDLine = ExternalModule(
    top="bitwiseANDLine",
    impl_path=f"{os.path.dirname(os.path.abspath(__file__))}/bitwiseAND.cc",
    input_idx=[0, 1],
    output_idx=[2],
)


def color_detect():
    ROW = 2

    @df.region()
    def top(Input: int8[ROW, 7680], Output: int8[ROW, 7680]):
        forward_input: Stream[int8[7680], 2][ROW]
        to_threshold0: Stream[int8[1920], 2][ROW]
        to_threshold1: Stream[int8[1920], 2][ROW]
        to_bitwise0: Stream[int8[1920], 2][ROW]
        to_bitwise1: Stream[int8[1920], 2][ROW]
        to_gray2RGBA: Stream[int8[1920], 2][ROW]
        to_and: Stream[int8[7680], 2][ROW]

        @df.kernel(mapping=[ROW], args=[Input])
        def RGBA2Hue(local_in: int8[ROW, 7680] @ [S(0), R]):
            pi = df.get_pid()
            to_threshold: int8[1920] = 0
            rgba2hueLine(local_in[0, :], to_threshold)
            to_threshold0[pi].put(to_threshold)
            to_threshold1[pi].put(to_threshold)
            forward_input[pi].put(local_in[0, :])

        @df.kernel(mapping=[ROW])
        def threshold0():
            pi = df.get_pid()
            threshold: int8[1920] = to_threshold0[pi].get()
            local: int8[1920] = 0
            thresholdLine40(threshold, local)
            to_bitwise: int8[1920] = 0
            thresholdLine30(local, to_bitwise)
            to_bitwise0[pi].put(to_bitwise)

        @df.kernel(mapping=[ROW])
        def threshold1():
            pi = df.get_pid()
            threshold: int8[1920] = to_threshold1[pi].get()
            local: int8[1920] = 0
            thresholdLine160(threshold, local)
            to_bitwise: int8[1920] = 0
            thresholdLine90(local, to_bitwise)
            to_bitwise1[pi].put(to_bitwise)

        @df.kernel(mapping=[ROW])
        def OR():
            pi = df.get_pid()
            local: int8[1920] = 0
            bitwiseORLine(to_bitwise0[pi].get(), to_bitwise1[pi].get(), local)
            to_gray2RGBA[pi].put(local)

        @df.kernel(mapping=[ROW])
        def Gray2RGBA():
            pi = df.get_pid()
            local: int8[7680] = 0
            gray2rgbaLine(to_gray2RGBA[pi].get(), local)
            to_and[pi].put(local)

        @df.kernel(mapping=[ROW], args=[Output])
        def AND(local_out: int8[ROW, 7680] @ [S(0), R]):
            pi = df.get_pid()
            bitwiseANDLine(to_and[pi].get(), forward_input[pi].get(), local_out[0, :])

    mod = df.build(top, target="aie")


if __name__ == "__main__":
    color_detect()
