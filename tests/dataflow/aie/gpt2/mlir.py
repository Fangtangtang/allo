import subprocess
import os
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16

np_supported_types = {
    "bf16": np.float32,  # numpy does not support bf16
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "ui1": np.bool_,
    "ui8": np.uint8,
    "ui16": np.uint16,
    "ui32": np.uint32,
    "ui64": np.uint64,
}


def read_tensor_from_file(dtype, shape, file_path):
    arr = np.fromfile(file_path, sep="\n", dtype=np_supported_types[str(dtype)])
    return arr.reshape(shape)


def call_mlir(
    project: str,
    dtype_list: list,
    trace_size: int,
    input_idx: list[int],
    output_idx: list[int],
    *args,
):
    # generate insts.txt
    cmd = f"cd {project} && aiecc.py --alloc-scheme=basic-sequential --aie-generate-xclbin --no-compile-host --xclbin-name=build/final.xclbin --no-xchesscc --no-xbridge --peano ${{PEANO_INSTALL_DIR}} --aie-generate-npu-insts --npu-insts-name=insts.txt top.mlir"
    with subprocess.Popen(cmd, shell=True) as process:
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to compile the MLIR-AIE code")
    cmd = f"cd {project}/build && cmake .. -DTARGET_NAME=top -DMLIR_AIE_DIR=$RUNTIME_LIB_DIR/.. && cmake --build . --config Release"
    with subprocess.Popen(cmd, shell=True) as process:
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to build AIE project.")
    # suppose the last argument is output
    for idx in input_idx:
        arg = args[idx]
        with open(
            os.path.join(project, f"input{idx}.data"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join([str(i) for i in arg.flatten()]))
    cmd = f"cd {project} && ./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE -p false --warmup 200 --test_iter 1000"
    with subprocess.Popen(cmd, shell=True) as process:
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to execute AIE code.")
    # for idx in output_idx:
    #     result = read_tensor_from_file(
    #         dtype_list[idx],
    #         args[idx].shape,
    #         f"{project}/output{idx}.data",
    #     )
    #     args[idx][:] = result


# fixme: update parameters as you need
from allo.ir.types import int8, int16, int32, bfloat16

N = 1024
D = 64

Q = np.random.randn(N, D).astype(np_bfloat16)
K = np.random.randn(N, D).astype(np_bfloat16)
V = np.random.randn(N, D).astype(np_bfloat16)
O = np.zeros(N * D).astype(np_bfloat16)
attention_score =np.random.randn(N, N).astype(np_bfloat16)
attn_weight = np.random.randn(N, N).astype(np_bfloat16)

# call_mlir(
#     "attn_score.prj",
#     [bfloat16, bfloat16, bfloat16],
#     0,
#     [0, 1],
#     [2],
#     Q,
#     K,
#     attention_score,
# )
call_mlir(
    "softmax.prj",
    [bfloat16, bfloat16],
    0,
    [0],
    [1],
    attention_score,
    attn_weight,
)
# call_mlir(
#     "gemm.prj",
#     [bfloat16, bfloat16],
#     0,
#     [0, 1],
#     [2],
#     attn_weight,
#     V, O
#     )

print("PASSED!")