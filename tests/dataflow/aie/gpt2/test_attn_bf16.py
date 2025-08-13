import os
import allo
from allo.ir.types import bfloat16
import allo.dataflow as df
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.experimental.external_kernel import ExternalModule

KERNEL_LIB_PATH = "/home/sf668/workspace/allo/tests/dataflow/aie/gpt2/kernels/"
np.random.seed(42)


# ################################################################
# Components
# ################################################################
def test_softmax(Q_slice=32, K_slice=32):
    # [NOTE]: Invalid compute kernel core_0, port number exceeded.
    softmax_kernel = ExternalModule(
        top="online_softmax",
        impl_path=KERNEL_LIB_PATH + "softmax.cc",
        input_idx=[0, 1, 2],
        output_idx=[3, 4, 5],
    )

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(
            attention_score: bfloat16[Q_slice, K_slice],
            prev_max_logit: bfloat16[Q_slice],
            prev_sum_exp: bfloat16[Q_slice],
            attention_weight: bfloat16[Q_slice, K_slice],
            new_max_logit: bfloat16[Q_slice],
            new_sum_exp: bfloat16[Q_slice],
        ):
            softmax_kernel(
                attention_score,
                prev_max_logit,
                prev_sum_exp,
                attention_weight,
                new_max_logit,
                new_sum_exp,
            )

    attention_score = np.random.randn(Q_slice, K_slice).astype(np_bfloat16)
    prev_max_logit = np.random.randn(Q_slice).astype(np_bfloat16)
    prev_sum_exp = np.random.randn(Q_slice).astype(np_bfloat16)
    attention_weight = np.zeros(Q_slice * K_slice).astype(np.float32)
    new_max_logit = np.zeros(Q_slice).astype(np.float32)
    new_sum_exp = np.zeros(Q_slice).astype(np.float32)
    mod = df.build(
        top,
        target="aie-mlir",
        profile=False,
    )
    mod(
        attention_score,
        prev_max_logit,
        prev_sum_exp,
        attention_weight,
        new_max_logit,
        new_sum_exp,
    )


# ################################################################
# Flash Attention
# ################################################################


def flash_attention(Q, K, V, chunk_size=32):
    """
    Single-batch FlashAttention

    Args: (N: sequence length, D: head dim)
        Q: (N, D) - Queries
        K: (N, D) - Keys
        V: (N, D) - Values
        chunk_size: int - how many queries to process at a time
    Returns:
        Output: (N, D)
    """
    N, D = K.shape
    output = np.zeros((N, D), dtype=Q.dtype)

    for q_start in range(0, N, chunk_size):
        q_end = min(q_start + chunk_size, N)
        Q_chunk = Q[q_start:q_end, :]  # (cq, D)

        # Initialize numerically stable softmax components
        max_logit = np.full((Q_chunk.shape[0], 1), -np.inf)
        sum_exp = np.zeros((Q_chunk.shape[0], 1))
        acc = np.zeros((Q_chunk.shape[0], D))

        for k_start in range(0, N, chunk_size):
            k_end = min(k_start + chunk_size, N)

            K_chunk = K[k_start:k_end, :]  # (ck, D)
            logits = Q_chunk @ K_chunk.T / np.sqrt(D)  # (cq, ck)
            local_max = np.max(logits, axis=1, keepdims=True)
            max_logit = np.maximum(max_logit, local_max)
            exp_logits = np.exp(logits - max_logit)

            V_chunk = V[k_start:k_end, :]  # (ck, D)
            sum_exp += exp_logits.sum(axis=1, keepdims=True)
            acc += exp_logits @ V_chunk
        output[q_start:q_end, :] = acc / sum_exp
        break

    return output


def gen_bundle(prefix, total):
    nodes = [f"{prefix}_{i}" for i in range(total)]
    return ("bundle", nodes)


def test_flash_attention(SEQ_LEN, HEAD_DIM, chunk_size):
    attn_score = ExternalModule(
        top="transpose_matmul_with_scale",
        impl_path=KERNEL_LIB_PATH + "attn_score.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    init_softmax = ExternalModule(
        top="init_softmax",
        impl_path=KERNEL_LIB_PATH + "softmax.cc",
        input_idx=[],
        output_idx=[0, 1],
    )

    online_softmax = ExternalModule(
        top="online_softmax",
        impl_path=KERNEL_LIB_PATH + "softmax.cc",
        input_idx=[0, 1, 2],
        output_idx=[3, 4, 5],
    )

    scale_attn_output = ExternalModule(
        top="scale_attn_output",
        impl_path=KERNEL_LIB_PATH + "attn_out.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = bfloat16
    Ly = Layout("S0R")

    @df.region()
    def top():
        q_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, HEAD_DIM), depth=2),
            shape=(SEQ_LEN // chunk_size,),
        )

        score_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, chunk_size), depth=2),
            shape=(SEQ_LEN // chunk_size,),
        )

        weight_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, chunk_size), depth=2),
            shape=(SEQ_LEN // chunk_size,),
        )

        o_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, HEAD_DIM), depth=2),
            shape=(SEQ_LEN // chunk_size,),
        )

        exp_sum_pipe = df.pipe(dtype=Ty, shape=(chunk_size,), depth=2)

        @df.kernel(mapping=[1])
        def send_q(Q: Ty[chunk_size, HEAD_DIM]):
            for i in range(SEQ_LEN // chunk_size):
                q_pipe[i].put(Q)

        @df.kernel(mapping=[SEQ_LEN // chunk_size])
        def cal_attn_score(K: Ty[SEQ_LEN, HEAD_DIM] @ Ly):
            score: Ty[chunk_size, chunk_size]
            pi = df.get_pid()
            attn_score(q_pipe[pi].get(), K, score)
            score_pipe[pi].put(score)

        @df.kernel(mapping=[1])
        def cal_softmax():
            max_logit: Ty[chunk_size]
            sum_exp: Ty[chunk_size]
            init_softmax(max_logit, sum_exp)
            # softmax
            for i in range(SEQ_LEN // chunk_size):
                attn_weight: Ty[chunk_size, chunk_size]
                online_softmax(
                    score_pipe[i].get(),
                    max_logit,
                    sum_exp,
                    attn_weight,
                    max_logit,
                    sum_exp,
                )
                weight_pipe[i].put(attn_weight)
            exp_sum_pipe.put(sum_exp)

        @df.kernel(mapping=[SEQ_LEN // chunk_size])
        def attn(V: Ty[SEQ_LEN, HEAD_DIM] @ Ly):
            pi = df.get_pid()
            o_pipe[pi].put(allo.matmul(weight_pipe[pi].get(), V))

        @df.kernel(mapping=[1])
        def acc(O: Ty[chunk_size, HEAD_DIM]):
            attn_output: Ty[chunk_size, HEAD_DIM] = 0
            for i in range(SEQ_LEN // chunk_size):
                attn_output[:, :] = allo.add(attn_output, o_pipe[i].get())
            scale_attn_output(attn_output, exp_sum_pipe.get(), O)

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[
            gen_bundle("cal_attn_score", SEQ_LEN // chunk_size),
            gen_bundle("attn", SEQ_LEN // chunk_size),
            ("chain", ["send_q_0", "cal_attn_score_0"]),
        ],
        profile=True,
        warmup=20,
        num_iters=100,
        device_type="npu1_2col",
    )
    Q = np.random.randn(chunk_size, D).astype(np_bfloat16)
    K = np.random.randn(N, D).astype(np_bfloat16)
    V = np.random.randn(N, D).astype(np_bfloat16)
    O = np.zeros(chunk_size * D).astype(np_bfloat16)
    mod(Q, K, V, O)
    out = flash_attention(Q, K, V, chunk_size=32)
    # print(out[:chunk_size])
    O = O.astype(np.float32).reshape(chunk_size, D)
    # print(O)


def test_temporal_attention(SEQ_LEN, HEAD_DIM, chunk_size):
    Q = np.random.randn(chunk_size, D).astype(np_bfloat16)
    K = np.random.randn(N, D).astype(np_bfloat16)
    V = np.random.randn(N, D).astype(np_bfloat16)
    O = np.zeros(chunk_size * D).astype(np_bfloat16)

    Ty = bfloat16

    # attn score
    attn_score = ExternalModule(
        top="transpose_matmul_with_scale",
        impl_path=KERNEL_LIB_PATH + "attn_score.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )
    ATTN_P0 = 1
    ATTN_P1 = 8
    ATTN_SCORE_M_TILE = ATTN_P0 * 32
    ATTN_SCORE_N_TILE = ATTN_P1 * 32
    ATTN_SCORE_LyA = Layout("S0R")
    ATTN_SCORE_LyB = Layout("S1R")
    ATTN_SCORE_LyC = Layout("S0S1")

    @df.region()
    def attn_score_kernel():
        @df.kernel(mapping=[ATTN_P1, ATTN_P0])
        def core(
            A: Ty[ATTN_SCORE_M_TILE, HEAD_DIM] @ ATTN_SCORE_LyA,
            B: Ty[ATTN_SCORE_N_TILE, HEAD_DIM] @ ATTN_SCORE_LyB,
            C: Ty[ATTN_SCORE_M_TILE, ATTN_SCORE_N_TILE] @ ATTN_SCORE_LyC,
        ):
            attn_score(A, B, C)

    attn_score_mod = df.build(
        attn_score_kernel, target="aie-mlir", project="attn_score.prj"
    )
    attention_score = np.empty((chunk_size, SEQ_LEN), dtype=np_bfloat16)
    for i in range(chunk_size // ATTN_SCORE_M_TILE):
        for j in range(SEQ_LEN // ATTN_SCORE_N_TILE):
            attn_score_mod(
                Q[
                    i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE,
                    :,
                ],
                K[
                    j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE,
                    :,
                ],
                attention_score[
                    i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE,
                    j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE,
                ],
            )
    # softmax
    softmax = ExternalModule(
        top="softmax_bfloat16",
        impl_path=KERNEL_LIB_PATH + "full_softmax.cc",
        input_idx=[0],
        output_idx=[1],
    )
    SOFTMAX_P0 = 4
    SOFTMAX_SEQ_TILE = 2 * SOFTMAX_P0
    SOFTMAX_Ly = Layout("S0R")

    @df.region()
    def softmax_kernel():
        @df.kernel(mapping=[SOFTMAX_P0])
        def core(
            input_x: Ty[SOFTMAX_SEQ_TILE, SEQ_LEN] @ SOFTMAX_Ly,
            output_x: Ty[SOFTMAX_SEQ_TILE, SEQ_LEN] @ SOFTMAX_Ly,
        ):
            softmax(input_x, output_x)

    softmax_mod = df.build(softmax_kernel, target="aie-mlir", project="softmax.prj")
    attn_weight = np.zeros((chunk_size, SEQ_LEN)).astype(np_bfloat16)
    for i in range(chunk_size // SOFTMAX_SEQ_TILE):
        softmax_mod(
            attention_score[i * SOFTMAX_SEQ_TILE : (i + 1) * SOFTMAX_SEQ_TILE, :],
            attn_weight[i * SOFTMAX_SEQ_TILE : (i + 1) * SOFTMAX_SEQ_TILE, :],
        )
    # PV
    attn_value = np.zeros((chunk_size, HEAD_DIM)).astype(np_bfloat16)
    LINEAR_M, LINEAR_N, LINEAR_K = 32, 64, 64
    linear_A_layout = Layout("S0R")
    linear_B_layout = Layout("RS1")
    linear_C_layout = Layout("S0S1")

    @df.region()
    def linear_matmul_kernel():
        @df.kernel(mapping=[4, 2])
        def gemm(
            A: Ty[LINEAR_M, LINEAR_K] @ linear_A_layout,
            B: Ty[LINEAR_K, LINEAR_N] @ linear_B_layout,
            C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        ):
            C[:, :] = allo.matmul(A, B)

    @df.region()
    def linear_accumulate_kernel():
        @df.kernel(mapping=[2, 4])
        def core(
            A: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
            B: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
            C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        ):
            C[:, :] = allo.add(A, B)

    linear_matmul_mod = df.build(
        linear_matmul_kernel, target="aie-mlir", project="linear_matmul.prj"
    )
    linear_accumulate_mod = df.build(
        linear_accumulate_kernel, target="aie-mlir", project="linear_accumulate.prj"
    )
    for j in range(HEAD_DIM // LINEAR_N):
        C_tmp = np.zeros((LINEAR_M, LINEAR_N)).astype(np_bfloat16)
        for k in range(SEQ_LEN // LINEAR_K):
            tile_A = attn_weight[
                :,
                k * LINEAR_K : (k + 1) * LINEAR_K,
            ]
            tile_B = V[
                k * LINEAR_K : (k + 1) * LINEAR_K,
                j * LINEAR_N : (j + 1) * LINEAR_N,
            ]
            linear_matmul_mod(tile_A, tile_B, C_tmp)
            linear_accumulate_mod(
                attn_value[
                    :,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ],
                C_tmp,
                attn_value[
                    :,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ],
            )
    return attn_value


if __name__ == "__main__":
    N, D = 2048, 64  # Sequence Length, Embedding Dim = 64
    chunk_size = 32
    # print(out.shape)
    # print(out)

    test_flash_attention(N, D, chunk_size)
    # test_temporal_attention(N, D, chunk_size)
