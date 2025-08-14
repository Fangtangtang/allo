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

    return output


def gen_bundle(prefix, idx, total):
    nodes = [f"{prefix}_{idx}_{i}" for i in range(total)]
    return ("bundle", nodes)


def test_flash_attention(SEQ_LEN, HEAD_DIM, chunk_size):
    iteration = SEQ_LEN // chunk_size
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
    Ly_outer = Layout("S1R")
    Ly_inner = Layout("S0R")

    @df.region()
    def top():
        q_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, HEAD_DIM), depth=2),
            shape=(SEQ_LEN // chunk_size, SEQ_LEN // chunk_size),
        )

        score_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, chunk_size), depth=2),
            shape=(SEQ_LEN // chunk_size, SEQ_LEN // chunk_size),
        )

        weight_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, chunk_size), depth=2),
            shape=(SEQ_LEN // chunk_size, SEQ_LEN // chunk_size),
        )

        o_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, HEAD_DIM), depth=2),
            shape=(SEQ_LEN // chunk_size, SEQ_LEN // chunk_size),
        )

        exp_sum_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size,), depth=2),
            shape=(SEQ_LEN // chunk_size,),
        )

        @df.kernel(mapping=[SEQ_LEN // chunk_size, 1])
        def send_q(Q: Ty[SEQ_LEN, HEAD_DIM] @ Ly_outer):
            po, _ = df.get_pid()
            for i in range(SEQ_LEN // chunk_size):
                q_pipe[po, i].put(Q)

        @df.kernel(mapping=[SEQ_LEN // chunk_size, SEQ_LEN // chunk_size])
        def cal_attn_score(K: Ty[SEQ_LEN, HEAD_DIM] @ Ly_inner):
            score: Ty[chunk_size, chunk_size]
            po, pi = df.get_pid()
            attn_score(q_pipe[po, pi].get(), K, score)
            score_pipe[po, pi].put(score)

        @df.kernel(mapping=[SEQ_LEN // chunk_size, 1])
        def cal_softmax():
            po, _ = df.get_pid()
            max_logit: Ty[chunk_size]
            sum_exp: Ty[chunk_size]
            init_softmax(max_logit, sum_exp)
            # softmax
            for i in range(SEQ_LEN // chunk_size):
                attn_weight: Ty[chunk_size, chunk_size]
                online_softmax(
                    score_pipe[po, i].get(),
                    max_logit,
                    sum_exp,
                    attn_weight,
                    max_logit,
                    sum_exp,
                )
                weight_pipe[po, i].put(attn_weight)
            exp_sum_pipe[po].put(sum_exp)

        @df.kernel(mapping=[SEQ_LEN // chunk_size, SEQ_LEN // chunk_size])
        def attn(V: Ty[SEQ_LEN, HEAD_DIM] @ Ly_inner):
            po, pi = df.get_pid()
            o_pipe[po, pi].put(allo.matmul(weight_pipe[po, pi].get(), V))

        @df.kernel(mapping=[SEQ_LEN // chunk_size, 1])
        def acc(O: Ty[SEQ_LEN, HEAD_DIM] @ Ly_outer):
            attn_output: Ty[chunk_size, HEAD_DIM] = 0
            po, _ = df.get_pid()
            for i in range(SEQ_LEN // chunk_size):
                attn_output[:, :] = allo.add(attn_output, o_pipe[po, i].get())
            scale_attn_output(attn_output, exp_sum_pipe[po].get(), O)

    mapping_primitives_ = []
    for idx in range(iteration):
        nodes = [f"cal_attn_score_{idx}_{i}" for i in range(SEQ_LEN // chunk_size)]
        mapping_primitives_.append(("bundle", nodes))
        nodes = [f"attn_{idx}_{i}" for i in range(SEQ_LEN // chunk_size)]
        mapping_primitives_.append(("bundle", nodes))
        mapping_primitives_.append(
            ("chain", [f"send_q_{idx}_0", f"cal_attn_score_{idx}_0"])
        )
    if iteration // 4 > 1:
        for i_ in range(4):
            mapping_primitives_.append(
                (
                    "bundle",
                    [
                        f"send_q_{idx*4+i_}_0-cal_attn_score_{idx*4+i_}_0"
                        for idx in range(iteration // 4)
                    ],
                )
            )
            mapping_primitives_.append(
                (
                    "bundle",
                    [f"cal_softmax_{idx*4+i_}_0" for idx in range(iteration // 4)],
                )
            )
            mapping_primitives_.append(
                ("bundle", [f"attn_{idx*4+i_}_0" for idx in range(iteration // 4)])
            )
            mapping_primitives_.append(
                ("bundle", [f"acc_{idx*4+i_}_0" for idx in range(iteration // 4)])
            )

    # print(mapping_primitives)
    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=mapping_primitives_,
        # mapping_primitives=[
        #     gen_bundle("cal_attn_score",0, SEQ_LEN // chunk_size),
        #     gen_bundle("attn",0, SEQ_LEN // chunk_size),
        #     ("chain", ["send_q_0_0", "cal_attn_score_0_0"]),
        #      gen_bundle("cal_attn_score",1, SEQ_LEN // chunk_size),
        #     gen_bundle("attn",1, SEQ_LEN // chunk_size),
        #     ("chain", ["send_q_1_0", "cal_attn_score_1_0"]),
        #     # ("bundle", ["send_q_0_0-cal_attn_score_0_0", "send_q_1_0-cal_attn_score_1_0"]),
        #     # ("bundle", ["cal_softmax_0_0", "cal_softmax_1_0"]),
        #     # ("bundle", ["attn_0_0", "attn_1_0"]),
        #     # ("bundle", ["acc_0_0", "acc_1_0"]),
        # ],
        profile=False,
        warmup=20,
        num_iters=100,
        # device_type="npu1_2col",
    )
    Q = np.random.randn(N, D).astype(np_bfloat16)
    K = np.random.randn(N, D).astype(np_bfloat16)
    V = np.random.randn(N, D).astype(np_bfloat16)
    O = np.zeros(N * D).astype(np_bfloat16)
    mod(Q, K, V, O)
    np.set_printoptions(threshold=np.inf)
    out = flash_attention(Q, K, V, chunk_size=32)
    print(out)
    O = O.astype(np.float32).reshape(N, D)
    print(O)


if __name__ == "__main__":
    N, D = 256, 64  # Sequence Length, Embedding Dim = 64
    chunk_size = 32
    # print(out.shape)
    # print(out)

    test_flash_attention(N, D, chunk_size)
