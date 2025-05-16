<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Experimental MLIR-AIE Codegen
## Environment Setup
Please follow the [Getting Started](https://github.com/Xilinx/mlir-aie/tree/main?tab=readme-ov-file#getting-started-for-amd-ryzen-ai-on-linux) guide to install MLIR-AIE.

In **Step 3: Install IRON library, mlir-aie, and llvm-aie compilers from wheels**, under the section [Install IRON for AMD Ryzen™ AI AIE Application Development](https://github.com/Xilinx/mlir-aie/tree/main?tab=readme-ov-file#install-iron-for-amd-ryzen-ai-aie-application-development), please install version `v1.0` using the following commands:
```bash
# Install IRON library and mlir-aie from a wheel
python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.0

# Install Peano from a llvm-aie wheel
python3 -m pip install https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025041501+b2a279c1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

Then, install Allo as usual:
```bash
git clone https://github.com/cornell-zhang/allo.git && cd allo
python3 -m pip install -v -e .
```

### Patches and Configuration
We rely on components from the [MLIR-AIE toolchain](https://github.com/Xilinx/mlir-aie) as libraries:

- To use [external kernels](https://github.com/Xilinx/mlir-aie/tree/ea9b4dfe7ea91f09c5c29c4d51ca74baea2dc4aa/aie_kernels) as an AIE kernel library, copy the directory to a desired location and set the environment variable:

  ```bash
  export MLIR_AIE_EXTERNAL_KERNEL_DIR=/your/copied/path/aie_kernels
  ```

- To use [runtime\_lib](https://github.com/Xilinx/mlir-aie/tree/ea9b4dfe7ea91f09c5c29c4d51ca74baea2dc4aa/runtime_lib) for the host, copy it to a desired location and set the environment variable:

  ```bash
  export RUNTIME_LIB_DIR=/your/copied/path/runtime_lib
  ```

When using `aiecc.py` to compile, we met various problems. 

If you encounter errors similar to 
```text
error: expected ')' at end of argument list
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
                                                          ^
``` 
To resolve this, you can patch the `downgrade_ir_for_peano` function in:
```text
$MLIR_AIE_INSTALL_DIR/python/aie/compiler/aiecc/main.py
```

Update the function as follows:

**Before:**

```python
def downgrade_ir_for_peano(llvmir):
    llvmir = llvmir.replace("getelementptr inbounds nuw", "getelementptr inbounds")
    return llvmir
```

**After:**

```python
def downgrade_ir_for_peano(llvmir):
    llvmir = llvmir.replace("getelementptr inbounds nuw", "getelementptr inbounds")
    llvmir = llvmir.replace("captures(none)", "")
    return llvmir
```

## Usage

To enable the experimental MLIR-AIE codegen, set the following environment variable:

```bash
export USE_AIE_MLIR_BUILDER=1
```

Then, specify `"aie-mlir"` as the target in the `dataflow.build` function.

### Example
vector addition
```python
import os
import allo
from allo.ir.types import int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

Ly = Layout("S0")


def _test_vector_scalar_add():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M]):
            B[:] = allo.add(A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        B = np.zeros(M).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
```

matrix multiplication
```python
import allo
from allo.ir.types import int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

LyA = Layout("S0R")
LyB = Layout("RS1")
LyC = Layout("S0S1")


def _test_gemm_1D():
    Ty = int32
    M, N, K = 16, 16, 16
    P0 = 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N], C: Ty[M, N] @ LyA):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(top, target="aie-mlir")
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")
```

producer consumer
```python
import os
import allo
from allo.ir.types import int32
import allo.dataflow as df
import numpy as np

Ty = int32
M, N, K = 16, 16, 16


@df.region()
def top():
    pipe = df.pipe(dtype=Ty, shape=(), depth=4)

    @df.kernel(mapping=[1])
    def producer(A: Ty[M, N]):
        for i, j in allo.grid(M, N):
            # load data
            out: Ty = A[i, j]
            # send data
            pipe.put(out)

    @df.kernel(mapping=[1])
    def consumer(B: Ty[M, N]):
        for i, j in allo.grid(M, N):
            # receive data
            data = pipe.get()
            # computation
            B[i, j] = data + 1


def test_producer_consumer():
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.zeros((M, N), dtype=np.int32)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        mod(A, B)
        np.testing.assert_allclose(A + 1, B, atol=1e-5)
        print("Passed!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

```

## New Feature
### Support for user-defined external kernels
Originally, complex computations on AIE cores were implemented using a limited set of [external kernels provided in the `mlir-aie` repository](https://github.com/Xilinx/mlir-aie/tree/ea9b4dfe7ea91f09c5c29c4d51ca74baea2dc4aa/aie_kernels). However, this external kernel library supports only a narrow range of operations and leaves room for performance improvement. To address these limitations, we add support for user-defined external kernels.

Users can now register and invoke external kernels implemented in C++ and exposed via extern "C" interfaces. These kernels can be written using the AIE API and integrated into the programming model workflow.

Suppose the external kernel is implemented in the `norm.cc` file:
```cpp
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define NOCPP

#define EPS 1e-6f // epsilon

template <typename T_in, typename T_out, const int SEQ_LEN, const int HIDDEN>
void rms_norm_single_batch(T_in *input_tensor, T_in *weight,
                           T_out *output_tensor) {
  constexpr int vec_factor = 16;
  using vec_t = aie::vector<T_in, vec_factor>;
  event0();
  for (int iter = 0; iter < SEQ_LEN; iter++) {
    T_in *__restrict input_ptr = input_tensor;
    T_in *__restrict weight_ptr = weight;
    T_out *__restrict output_ptr = output_tensor;
    float square_sum = 0.0f;
    const int F = HIDDEN / vec_factor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      vec_t square_vec = aie::mul(input_vec, input_vec);
      square_sum += aie::reduce_add(square_vec);
    }
    vec_t square_sum_vec =
        aie::broadcast<T_in, vec_factor>(square_sum / HIDDEN + EPS);
    vec_t rms = aie::invsqrt(square_sum_vec);
    input_ptr = input_tensor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      vec_t normed = aie::mul(input_vec, rms);
      vec_t weight_vec = aie::load_v<vec_factor>(weight_ptr);
      weight_ptr += vec_factor;
      vec_t result = aie::mul(normed, weight_vec);
      aie::store_v(output_ptr, result);
      output_ptr += vec_factor;
    }
    input_tensor += HIDDEN;
    output_tensor += HIDDEN;
  }
  event1();
}
```
and exposed via extern "C" interfaces
```cpp
extern "C" {
  void layer_norm(float A_in[4][512], float B_in[512], float C_out[4][512]) {
    rms_norm_single_batch<float, float, 4, 512>(&A_in[0][0], B_in, &C_out[0][0]);
  }
}
```

We can create an [ExternalModule](external_kernel.py) to wrap the kernel and use it in computation on AIE core.

Register the `ExternalModule` in the context.
```python
norm = ExternalModule(
    top="layer_norm",       # Name of the top-level function defined with `extern "C"`
    impl_path="norm.cc",    # Path to the user-provided source file that implements the external kernel
    input_idx=[0, 1],       # Indices of input arguments in the argument list passed to the module
    output_idx=[2],         # Indices of output arguments in the argument list passed to the module
)
```
And the external module can then be used in an Allo kernel.
```python
@df.kernel(mapping=[1])
    def core(A: Ty[M, N] @ LyA, B: Ty[N] @ Ly, C: Ty[M, N] @ LyA):
        norm(A, B, C)
```

An example can be found in [`tests/dataflow/aie/customized_kernel`](../../../tests/dataflow/aie/customized_kernel/)