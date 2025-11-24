import allo
import json
import subprocess
from allo.ir.types import int8, int16, bfloat16
import allo.dataflow as df
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16

PRJ_DIR = "matmul.prj"


def _test_gemm_1D(M, N, K, Ty):

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def gemm(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(
        top, project=PRJ_DIR, target="aie", profile=True, trace=[("gemm", (0,))]
    )
    if Ty == int16:
        A = np.random.randint(0, 64, (M, K)).astype(np.int16)
        B = np.random.randint(0, 64, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
    elif Ty == bfloat16:
        A = (np.random.random((M, K)) * 0.1).astype(np_bfloat16)
        B = (np.random.random((K, N)) * 0.1).astype(np_bfloat16)
        C = np.zeros((M, N)).astype(np_bfloat16)
    mod(A, B, C)
    if Ty is bfloat16:
        np.testing.assert_allclose(
            C.astype(np.float32), (A @ B).astype(np.float32), atol=1e-1
        )
    else:
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    m, n, k = 16, 16, 8
    for M in range(m, 512, m):
        for N in range(n, 512, n):
            dead = 0
            for K in range(k, 512, k):
                try:
                    M, N, K = 64, 16, 56
                    _test_gemm_1D(M, N, K, bfloat16)
                    cmd = f"cd {PRJ_DIR} &&  ~/usr/mlir-aie/programming_examples/utils/parse_trace.py --filename trace.txt --mlir top.mlir --colshift 1 > trace.json"
                    with subprocess.Popen(cmd, shell=True) as process:
                        process.wait()
                    if process.returncode != 0:
                        raise RuntimeError("Failed to parse trace.")

                    with open(f"{PRJ_DIR}/trace.json", "r") as f:
                        data = json.load(f)
                    cycle_cnt = []
                    prev = -1
                    for event in data:
                        if "name" in event:
                            if event["name"] == "INSTR_EVENT_0" and event["ph"] == "E":
                                prev = int(event["ts"])
                            if event["name"] == "INSTR_EVENT_1" and event["ph"] == "B":
                                if prev > 0:
                                    cycle_cnt.append(int(event["ts"]) - prev)
                                    prev = -1
                    print(cycle_cnt)
                    avg = sum(cycle_cnt) / len(cycle_cnt)
                    with open("avg.txt", "a") as f:
                        f.write(f"{M}, {N}, {K}, {avg}, {(M*N*K/128)/avg}\n")
                except:
                    dead += 1
                    if dead > 3:
                        break
                    import sys

                    sys.exit(0)
                import sys

                sys.exit(0)
