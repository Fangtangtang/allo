<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# NPU Experiments

This directory contains reproducible experiment runners for comparing Allo and
handwritten mlir-aie implementations on a Ryzen AI NPU. The first experiment is
GEMM, driven by [`gemm.py`](gemm.py).

The default GEMM sweep covers every Cartesian product of
`M,N,K = 256,512,1024,2048` and the three same-width input/output datatypes
`int16`, `int8`, and `bf16`. This produces 192 configurations per flow or 384
configurations for `--flow both`.

## Environment setup

Pull and start the Allo AIE container with the NPU device mounted:

```bash
docker pull shihanfang/allo-ci:aie-v1.0
docker run --rm -it \
  --device /dev/accel/accel0:/dev/accel/accel0 \
  --ulimit memlock=-1 \
  -v "$(pwd):/ryzers/allo" \
  -w /ryzers/allo \
  shihanfang/allo-ci:aie-v1.0 bash
```

Inside the container, activate and build Allo:

```bash
conda activate allo
cd /ryzers/allo
rm -rf mlir/build
python3 -m pip install -v -e . --no-build-isolation
```

The runner checks for `/dev/accel/accel0`, the AIE environment variables, and
the required mlir-aie build tools before starting a hardware run. `list` and
`run --dry-run` do not require the NPU.

## Running GEMM experiments

Inspect the complete 384-case sweep without compiling anything:

```bash
python3 aie-experiments/gemm.py list --flow both
```

Run the complete sweep for both implementations:

```bash
python3 aie-experiments/gemm.py run --flow both
```

Run only the Allo int8 `256x256x256` smoke case with shorter timing settings:

```bash
python3 aie-experiments/gemm.py run \
  --flow allo \
  --dtype int8 \
  --M 256 --N 256 --K 256 \
  --warmup 1 --iterations 5
```

Run selected bf16 shapes through mlir-aie. Multi-value dimension flags form a
Cartesian product, so this command runs two cases:

```bash
python3 aie-experiments/gemm.py run \
  --flow mlir-aie \
  --dtype bf16 \
  --M 256 512 \
  --N 512 \
  --K 1024
```

Rerun the 32 Allo bf16 cases with `K=1024` or `K=2048` while retaining timing
data when only the output comparison fails:

```bash
python3 aie-experiments/gemm.py run \
  --flow allo \
  --dtype bf16 \
  --K 1024 2048 \
  --benchmark-on-validation-failure
```

The opt-in applies only to Allo bf16 comparison assertions. Such a case still
has `status: "failed"` and `validation: "failed"`, but records all timing
samples, sets `timed_validation_failure: true`, continues the sweep, and does
not make the command return nonzero. Build, runtime, device, and timing failures
remain hard failures. Existing untimed failed records are not resumable, so no
`--rerun` is needed for the first command above.

Select multiple datatypes or dimension values in the same way:

```bash
python3 aie-experiments/gemm.py run \
  --flow both \
  --dtype int16 int8 \
  --M 256 512 \
  --N 256 512 \
  --K 1024
```

Preview the resolved cases and principal build commands without executing them:

```bash
python3 aie-experiments/gemm.py run \
  --flow both \
  --dtype int8 \
  --M 256 --N 256 --K 256 \
  --dry-run
```

Preserve generated Allo projects, MLIR, xclbins, and executables for completed
cases:

```bash
python3 aie-experiments/gemm.py run --flow both --keep-builds
```

Force configurations with existing completed records to run again:

```bash
python3 aie-experiments/gemm.py run --flow both --rerun
```

Regenerate the raw, filtered, and summary CSV files without accessing hardware:

```bash
python3 aie-experiments/gemm.py process \
  --output-dir aie-experiments/results/gemm
```

## Plotting GEMM results

After completing and processing the default 384-case sweep, generate one
arithmetic-intensity plot for each datatype:

```bash
python3 aie-experiments/plot.py
```

By default, the plotter reads `aie-experiments/results/gemm/summary.csv` and
writes `gemm_int16.png`, `gemm_int8.png`, and `gemm_bf16.png` under the writable
`aie-experiments/plots` directory. Override these locations with `--summary`
and `--output-dir`.

Each plot contains Allo and MLIR-AIE best-performance envelopes. Arithmetic
intensity counts one read of each input and one output write:

```text
OPs/byte = 2 * M * N * K / (dtype_bytes * (M * K + K * N + M * N))
```

At each repeated arithmetic intensity, the plotter selects the configuration
with the highest `filtered_gflops` value and converts it to TOP/s. Shaded areas
show the best-performance envelopes derived from `filtered_min_us` and
`filtered_max_us` at the same arithmetic intensities. It validates each
datatype's 128-case sub-sweep independently. A datatype with missing, duplicate,
ordinary failed, unvalidated, unexpected, or non-positive results is skipped
with a warning, while complete datatype plots are still generated. A failed
Allo bf16 result with `timed_validation_failure: true` and complete filtered
metrics is accepted and plotted with the same curve and shaded-range style as a
successful result. The command fails without generating plots only when no
datatype has a complete sweep.

Use `--help` after either the top-level command or a subcommand for the current
CLI syntax:

```bash
python3 aie-experiments/gemm.py --help
python3 aie-experiments/gemm.py run --help
```

## Command flags

| Flag | Commands | Meaning |
| --- | --- | --- |
| `--flow {allo,mlir-aie,both}` | `run`, `list` | Required flow selection. `both` executes Allo cases first and mlir-aie cases second. |
| `--dtype TYPE [TYPE ...]` | `run`, `list` | One or more of `int16`, `int8`, and `bf16`. Defaults to all three. The output type is the same as the input type. |
| `--M SIZE [SIZE ...]` | `run`, `list` | One or more M dimensions. Defaults to `256 512 1024 2048`. |
| `--N SIZE [SIZE ...]` | `run`, `list` | One or more N dimensions. Defaults to `256 512 1024 2048`. |
| `--K SIZE [SIZE ...]` | `run`, `list` | One or more K dimensions. Defaults to `256 512 1024 2048`. |
| `--warmup COUNT` | `run` | Unrecorded warmup kernel launches per case. Defaults to `20`. |
| `--iterations COUNT` | `run` | Recorded kernel launches per case. Defaults to `200`. |
| `--benchmark-on-validation-failure` | `run` | For selected Allo bf16 cases only, continue timing after a comparison assertion, preserve failed status, and treat the complete timing record as expected. |
| `--output-dir PATH` | `run`, `process` | Result directory. Defaults to `aie-experiments/results/gemm`. |
| `--mlir-aie-root PATH` | `run` | mlir-aie checkout inside the image. Defaults to `/ryzers/mlir-aie`. |
| `--rerun` | `run` | Ignore matching completed records and execute those cases again. |
| `--keep-builds` | `run` | Keep completed per-case build directories. Hard-failed build directories are always retained. |
| `--fail-fast` | `run` | Stop after the first hard-failed configuration. Marked timed-validation failures still continue. |
| `--dry-run` | `run` | Print resolved cases and principal commands without checking the environment, creating results, compiling, or running the NPU. |

Repeated values are deduplicated. The `--M`, `--N`, `--K`, and `--dtype` values
are combined as a Cartesian product.

## Experiment behavior

Each configuration uses the following tile sizes:

| Datatype | m | n | k |
| --- | ---: | ---: | ---: |
| `int16` | 64 | 64 | 64 |
| `bf16` | 64 | 64 | 64 |
| `int8` | 64 | 128 | 64 |

Both flows target NPU1 and use four compute rows. The runner selects the largest
valid NPU1 column count for a configuration. All cases use four columns except
`int8` with `N=256`, which uses two columns because the mlir-aie whole-array
design requires `N` to be divisible by `n * n_aie_cols`. The same active column
count is passed to Allo for a comparable mapping.

For each case, the runner:

1. Builds the implementation in an isolated work directory.
2. Performs one correctness run outside the timing sample set.
3. Performs the requested warmups.
4. Records every device kernel launch-to-wait time in microseconds.
5. Requires exactly `--iterations` timing samples.
6. Writes an atomic per-case JSON result and regenerates aggregate CSV files at
   the end of the command.

Integer Allo results require exact agreement with NumPy. Allo bf16 uses the
same `atol=1e-1` check as the repository GEMM example. With the opt-in flag, a
failed bf16 comparison writes the NumPy mismatch traceback and the
machine-readable `ALLO_VALIDATION=FAILED` marker to the case log before warmup
and timed iterations continue. The upstream mlir-aie host remains strict and
performs its built-in deterministic verification; large matrices use its
stochastic verification path. Verification is disabled only during the timed
mlir-aie loop.

Host setup, xclbin loading, buffer synchronization, compilation, and reference
checking are excluded from each recorded device time. Per-sample throughput is
calculated as:

```text
GFLOP/s = 2 * M * N * K / (time_us * 1000)
```

## Resume and failure handling

A successful case is skipped when its saved signature exactly matches the
requested flow, datatype, dimensions, tile sizes, NPU target, warmup count, and
iteration count, and its JSON record contains the expected number of samples.
A marked timed-validation failure is also skipped when the same opt-in flag and
complete sample count match. The opt-in field appears in a signature only when
enabled, preserving existing successful signatures. Use `--rerun` to replace
either kind of completed result. Changing timing settings also reruns the case.

By default, a failed case is recorded and the sweep continues. Its complete
build/runtime output and Python traceback are kept in the case log, and its work
directory is retained for debugging. `--fail-fast` stops after the first hard
failure. An interrupted case is marked failed and runs again with the same
command.

A marked timed-validation failure is expected completed work: it does not
trigger `--fail-fast`, does not contribute to a nonzero exit status, and its
build directory is deleted unless `--keep-builds` is set. Any other compile,
runtime, device, sample-count, or timing failure remains hard and returns
nonzero.

Completed work directories are deleted to control disk usage. Use
`--keep-builds` when generated MLIR, xclbins, executables, or Allo projects are
needed for inspection.

## Results and filtering

The default result layout is:

```text
aie-experiments/results/gemm/
├── cases/
│   ├── allo/<dtype>/<case>.json
│   └── mlir-aie/<dtype>/<case>.json
├── logs/
│   ├── allo/<dtype>/<case>.log
│   └── mlir-aie/<dtype>/<case>.log
├── raw_timings.csv
├── filtered_timings.csv
└── summary.csv
```

The per-case JSON files are the resumable source of truth. They contain the
configuration, status, validation result, exact commands, timestamps, all raw
timings, error text, and log location.

`raw_timings.csv` contains one row per captured sample with the case metadata,
sample index, `time_us`, calculated `gflops`, `is_outlier`, validation, and
`timed_validation_failure` fields. `filtered_timings.csv` contains retained
samples from successful cases and marked timed-validation failures; ordinary
failed records remain raw-only. `summary.csv` contains one row per case with
status, validation, and `timed_validation_failure` fields, raw/filtered sample
counts, mean, median, minimum, maximum, population standard deviation, GFLOP/s
based on mean time, quartiles, filter bounds, elapsed wall time, errors, and log
path.

Filtering uses NumPy linear quartiles and the inclusive Tukey fence:

```text
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
keep lower <= time_us <= upper
```

No raw timing is discarded or overwritten. If `IQR` is zero, values equal to
the quartiles are retained and different values fall outside the zero-width
fence. Run the `process` command at any time to reproduce the filtered and
summary tables from the JSON records.

## Partial-NPU bf16 GEMM experiment

[`gemm_partial_npu.py`](gemm_partial_npu.py) runs the same 64-shape bf16
Cartesian sweep with 1x4, 2x4, and 4x4 logical compute-tile configurations.
The three logical variants are:

| Variant | Mapping shape | Device width |
| --- | ---: | ---: |
| `manual` (Manual Template) | mlir-aie `n_aie_cols=1/2/4` | 1/2/4 columns |
| `compiled` (Compiled) | Allo 2x2 / 4x2 / 4x4 | `npu1_1col` / `npu1_2col` / `npu1_4col` |
| `compiled-full-io` (Compiled (Full I/O)) | Allo 4x1 / 4x2 / 4x4 | `npu1_4col` |

Inspect the default expansion without accessing hardware:

```bash
python3 aie-experiments/gemm_partial_npu.py list
```

The default expansion contains 512 physical runs and 576 logical plot points.
At 4x4, the two Allo configurations have the same mapping and device, so the
runner executes one canonical `compiled` case and records that it supplies both
Allo plot series.

The 1x4 logical `compiled` configuration is intentionally generated with
`row_num=2` and `col_num=2` in the Allo mapping primitives while retaining
`device_type=npu1_1col`. The other configurations use four mapping rows.

Run and process the complete experiment:

```bash
python3 aie-experiments/gemm_partial_npu.py run
python3 aie-experiments/gemm_partial_npu.py process
```

Results default to `aie-experiments/results/gemm-partial-npu`. The runner
supports `--variant`, `--columns`, `--M`, `--N`, and `--K` selections,
plus `--warmup`, `--iterations`, `--output-dir`, `--mlir-aie-root`,
`--rerun`, `--keep-builds`, `--fail-fast`, and `--dry-run`. For example,
preview the two-column commands for one shape with:

```bash
python3 aie-experiments/gemm_partial_npu.py run \
  --columns 2 \
  --M 256 --N 256 --K 256 \
  --dry-run
```

All Allo cases automatically continue through warmup and timing if the bf16
output comparison fails. The JSON and CSV rows retain `status: "failed"`,
`validation: "failed"`, the validation traceback in the case log, all timing
samples, and `timed_validation_failure: true`. These marked cases are
filterable, resumable with matching timing settings and sample counts, cleaned
up unless `--keep-builds` is used, and do not cause a nonzero exit status.
Compilation, runtime, device, or timing failures remain hard failures. The
manual-template validation path remains strict.

The Full I/O variant restricts the Allo mapping to the requested logical
compute width but targets `npu1_4col`, exposing all four memory and interface
tiles. The existing placer may distribute the logical compute nodes across that
physical four-column mesh.

After the selected complete sweep is processed, generate the three plots:

```bash
python3 aie-experiments/plot_partial_npu.py
```

The plotter reads
`aie-experiments/results/gemm-partial-npu/summary.csv` by default and writes
`gemm_bf16_1x4.png`, `gemm_bf16_2x4.png`, and `gemm_bf16_4x4.png` to
`aie-experiments/plots`. Each plot contains the three logical series, uses
OPs/byte and TOP/s units, and includes Tukey-filtered best-performance
envelopes with min/max shaded ranges. At 4x4, the green dashed Full I/O curve
and band reuse the canonical red Compiled data and therefore overlap it.

Completeness is validated independently for each compute width. A width with a
missing, duplicate, hard-failed, or invalid row is skipped without blocking
complete widths; no PNGs are written when no width is complete. Marked timed
Allo validation failures with complete filtered metrics are accepted as normal
performance data.
