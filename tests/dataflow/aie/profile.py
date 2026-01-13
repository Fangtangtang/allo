# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import allo
import allo.dataflow as df
from allo.ir.types import int8, int32, bfloat16, Stream
from allo.memory import Layout
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
from allo.backend.aie.external_kernel import ExternalModule

# RRxRS->RS
# RSxSR->RR
LyW1 = Layout("RS0")
LyW2 = Layout("S0R")

event0 = ExternalModule(
    top="mark_start",
    impl_path="../../../allo/library/aie/event.cc",
    input_idx=[],
    output_idx=[],
)

event1 = ExternalModule(
    top="mark_end",
    impl_path="../../../allo/library/aie/event.cc",
    input_idx=[],
    output_idx=[],
)

import os

bitwiseANDLine = ExternalModule(
    top="bitwiseANDLine",
    impl_path=f"{os.path.dirname(os.path.abspath(__file__))}/vision/bitwiseAND.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

scale_attn_output = ExternalModule(
    top="scale_attn_output",
    impl_path="/home/sf668/workspace/allo/allo/library/aie/kernels/attn_out.cc",
    input_idx=[0, 1],
    output_idx=[2],
)


def profiling():

    @df.region()
    def top(Input1: bfloat16[32], Output: bfloat16[32]):
        @df.kernel(mapping=[1], args=[Input1, Output])
        def gemm(inp1: bfloat16[32], outp: bfloat16[32]):
            attn_weight: bfloat16[32, 64] = 1
            outp = 1
            scale_attn_output(attn_weight, inp1, attn_weight)

    mod = df.build(
        top, target="aie", profile=True, trace=[("gemm", (0,))], trace_size=65536
    )
    A = (np.random.random((32)) * 0.1).astype(np_bfloat16)
    B = (np.random.random((32, 64)) * 0.1).astype(np_bfloat16)
    C = np.zeros((32, 64)).astype(np_bfloat16)
    # A = np.random.randint(64, 128, (7680,)).astype(np.int8)
    # B = np.random.randint(0, 64, (7680,)).astype(np.int8)
    # C = np.zeros((7680,)).astype(np.int8)
    mod(A, A)
    # np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def test():
    Ty = int32
    M, N = 6, 8
    P0 = 3

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(In: Ty[M, N], Out0: Ty[M, N], Out1: Ty[M, N], Out2: Ty[M, N]):
            pn = df.get_pid()
            with allo.meta_if(pn == 0):
                Out0[:, :] = In
            with allo.meta_elif(pn == 1):
                Out1[:, :] = allo.add(In, 1)
            with allo.meta_else():
                Out2[:, :] = allo.add(In, 2)

    A = np.random.randint(0, 100, (M, N)).astype(np.int32)

    mod = df.build(top, target="aie")
    B0 = np.zeros((M, N)).astype(np.int32)
    B1 = np.zeros((M, N)).astype(np.int32)
    B2 = np.zeros((M, N)).astype(np.int32)
    mod(A, B0, B1, B2)
    np.testing.assert_allclose(B0, A)
    np.testing.assert_allclose(B1, A + 1)
    np.testing.assert_allclose(B2, A + 2)
    print("PASSED!")


def parse_trace(project="top.prj"):
    cmd = f"~/usr/mlir-aie/programming_examples/utils/parse_trace.py --filename {project}/trace.txt --mlir {project}/top.mlir --colshift 1 > {project}/trace.json"
    with subprocess.Popen(cmd, shell=True) as process:
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to execute AIE code.")
    with open(f"{project}/trace.json") as f:
        trace = json.load(f)
    latency = []
    begin = -1
    for event in trace:
        if event["name"] == "INSTR_EVENT_0" and event["ph"] == "E" and begin < 0:
            begin = event["ts"]
        if event["name"] == "INSTR_EVENT_1" and event["ph"] == "B":
            if begin > 0:
                latency.append(event["ts"] - begin)
            begin = -1
    print(latency)
    print(sum(latency) / len(latency))
    import numpy as np

    lat = np.array(latency)

    q1 = np.percentile(lat, 25)
    q3 = np.percentile(lat, 75)
    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    filtered = lat[(lat >= low) & (lat <= high)]
    avg = filtered.mean()
    print(avg)


if __name__ == "__main__":
    profiling()
    parse_trace()
    # test()
