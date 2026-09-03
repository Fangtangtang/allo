<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# NPU Experiments

This directory contains reproducible experiment runners for comparing Allo and
handwritten mlir-aie implementations on Ryzen AI NPUs. Both runners and both
plotters accept `--device {xdna1,xdna2}`; XDNA1 remains the default. The first
experiment is GEMM, driven by [`gemm.py`](gemm.py).

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
`run --dry-run` do not require the NPU. The selected device is passed
explicitly to every backend worker; the runner sets `NPU2=0` for XDNA1 and
`NPU2=2` for XDNA2 instead of inferring a device from the ambient environment.

## Running GEMM experiments

Inspect the complete 384-case sweep without compiling anything:

```bash
python3 aie-experiments/gemm.py list --flow both
```

Run the complete XDNA1 sweep for both implementations:

```bash
python3 aie-experiments/gemm.py run --flow both
```

Select XDNA2 explicitly. This still contains 384 cases:

```bash
python3 aie-experiments/gemm.py run \
  --device xdna2 \
  --flow both \
  --mlir-aie-root /home/sf668/usr/mlir-aie \
  --benchmark-on-validation-failure
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

Preserve generated artifacts only for one selected diagnostic case:

```bash
python3 aie-experiments/gemm.py run \
  --flow allo --dtype int8 \
  --M 256 --N 256 --K 256 \
  --keep-builds --fail-fast
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
python3 aie-experiments/plot.py --device xdna2
```

XDNA1 reads `results/gemm/summary.csv` and writes to `plots/`. XDNA2 reads
`results/gemm-xdna2/summary.csv` and writes to `plots/xdna2/`. All paths are
relative to `aie-experiments/`; override them with `--summary` and
`--output-dir`. The plotter verifies the recorded device, backend target, and
effective `NPU2` provenance before accepting a row.

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
| `--device {xdna1,xdna2}` | `run`, `list`, `process`, plotters | Select the NPU generation and device-specific default paths. Defaults to `xdna1`. |
| `--flow {allo,mlir-aie,both}` | `run`, `list` | Required flow selection. `both` executes Allo cases first and mlir-aie cases second. |
| `--dtype TYPE [TYPE ...]` | `run`, `list` | One or more of `int16`, `int8`, and `bf16`. Defaults to all three. The output type is the same as the input type. |
| `--M SIZE [SIZE ...]` | `run`, `list` | One or more M dimensions. Defaults to `256 512 1024 2048`. |
| `--N SIZE [SIZE ...]` | `run`, `list` | One or more N dimensions. Defaults to `256 512 1024 2048`. |
| `--K SIZE [SIZE ...]` | `run`, `list` | One or more K dimensions. Defaults to `256 512 1024 2048`. |
| `--warmup COUNT` | `run` | Unrecorded warmup kernel launches per case. Defaults to `20`. |
| `--iterations COUNT` | `run` | Recorded kernel launches per case. Defaults to `200`. |
| `--benchmark-on-validation-failure` | `run` | For selected Allo bf16 cases only, continue timing after a comparison assertion, preserve failed status, and treat the complete timing record as expected. |
| `--output-dir PATH` | `run`, `process` | Result directory. Defaults to `results/gemm` on XDNA1 or `results/gemm-xdna2` on XDNA2. |
| `--mlir-aie-root PATH` | `run` | mlir-aie checkout inside the image. Defaults to `/ryzers/mlir-aie`. |
| `--rerun` | `run` | Ignore matching completed records and execute those cases again. |
| `--keep-builds` | `run` | Keep the clean per-case work directory after any outcome for single-case diagnosis. |
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

Both devices use four compute rows. Device behavior is centralized:

| Device | Compute tiles | Width candidates | `NPU2` | mlir-aie | Allo |
| --- | ---: | --- | ---: | --- | --- |
| XDNA1 | 4x4 | `4,2,1` | `0` | `devicename=npu` | `npu1_<N>col` |
| XDNA2 | 4x8 | `8,4,2,1` | `2` | `devicename=npu2` | `npu2_<N>col` for 1-7, `npu2` for 8 |

The main runner selects the largest width for which `N` is divisible by
`n * n_aie_cols`. Thus XDNA1 uses four columns except int8 at `N=256`,
which uses two. On XDNA2, int16/bf16 use four columns at `N=256` and eight
at larger N; int8 uses 2, 4, and 8 columns at `N=256`, `N=512`, and
`N>=1024`, respectively. Both backends receive the same active width.

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

Before every attempted case, the runner removes its stale work directory and
creates a clean one. Unless `--keep-builds` is set, the whole directory is
removed in a `finally` path after success, a timed validation failure, a hard
failure, or interruption. The sweep-scoped instrumented mlir-aie host cache is
also removed at the end even when the sweep fails. Logs and JSON records retain
the error or interruption details; `--fail-fast` stops after the first hard
failure.

Use `--keep-builds` only to diagnose one selected failing case, for example:

```bash
python3 aie-experiments/gemm.py run \
  --device xdna2 --flow allo --dtype int16 \
  --M 256 --N 512 --K 256 \
  --keep-builds --fail-fast --rerun
```

After diagnosis, rerun that case without `--keep-builds` (or remove its
device-scoped case directory). Production sweeps intentionally retain only
per-case JSON, logs, aggregate CSVs, and generated plots—not `.prj`, xclbin,
MLIR, object, or executable artifacts.

## Results and filtering

Device-specific result and plot roots are:

| Device | Main results | Partial results | Plots |
| --- | --- | --- | --- |
| XDNA1 | `results/gemm` | `results/gemm-partial-npu` | `plots` |
| XDNA2 | `results/gemm-xdna2` | `results/gemm-partial-npu-xdna2` | `plots/xdna2` |

All paths above are relative to `aie-experiments/`. Each result root has this
layout:

```text
<result-root>/
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
timings, error text, log location, selected `device`, backend `target`, exact
`backend_target`, and effective `npu2` value. The aggregate CSVs carry the
same provenance fields.

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

## End-to-end attention experiment

[`attention.py`](attention.py) compares the four-module unfused bf16 attention
baseline with the fused implementation from
[`examples/aie/attention.py`](../examples/aie/attention.py). Both use one head,
`HEAD_DIM=64`, and sequence lengths `64,128,256,512,1024,2048`. FlashAttention
uses 32x32 query/KV chunks. XDNA1 uses its 4x4 compute array and XDNA2 uses its
whole 4x8 compute array. The unfused baseline at sequence length 2048 is marked
infeasible and is not built or executed on either device.

### Docker setup

Run the experiment in the documented AIE image. From the repository root:

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
rm -rf mlir/build
python3 -m pip install -v -e . --no-build-isolation
```

Only `run` needs the mounted NPU. Listing cases, dry runs, result processing,
and plotting existing data do not access hardware.

### Commands and usage

List all twelve default cases for either device:

```bash
python3 aie-experiments/attention.py list
python3 aie-experiments/attention.py list --device xdna2
```

Preview commands without compiling or creating files, then run a short XDNA2
smoke benchmark:

```bash
python3 aie-experiments/attention.py run --device xdna2 --dry-run
python3 aie-experiments/attention.py run \
  --device xdna2 --implementation both --seq-len 64 \
  --warmup 1 --iterations 5
```

Run the complete XDNA1 or XDNA2 sweep:

```bash
python3 aie-experiments/attention.py run
python3 aie-experiments/attention.py run --device xdna2
```

Select an implementation and one or more sequence lengths:

```bash
python3 aie-experiments/attention.py run \
  --implementation baseline --seq-len 64 128 256
python3 aie-experiments/attention.py run \
  --device xdna2 --implementation flash --seq-len 512 1024 2048
```

Matching current-version successful records, timed baseline validation failures,
and the infeasible baseline/2048 record resume automatically. Runnable legacy
records without paired NPU samples are not resumable and must be measured again;
use `--rerun` to force replacement of selected records. Retain generated projects
and their `test.cpp` files with `--keep-builds`:

```bash
python3 aie-experiments/attention.py run \
  --device xdna2 --seq-len 1024 --rerun
python3 aie-experiments/attention.py run \
  --implementation both --seq-len 64 --keep-builds --rerun
```

Regenerate aggregate CSV files and create the device-specific bar plot:

```bash
python3 aie-experiments/attention.py process
python3 aie-experiments/attention.py process --device xdna2
python3 aie-experiments/plot_attention.py
python3 aie-experiments/plot_attention.py --device xdna2
```

Override both result and plot locations explicitly:

```bash
python3 aie-experiments/attention.py run \
  --seq-len 64 --output-dir /tmp/attention-results
python3 aie-experiments/attention.py process \
  --output-dir /tmp/attention-results
python3 aie-experiments/plot_attention.py \
  --summary /tmp/attention-results/summary.csv \
  --output-dir /tmp/attention-plots
```

XDNA1 results default to `results/attention`, while XDNA2 results default to
`results/attention-xdna2`. Each root contains resumable `cases/`, `logs/`,
`raw_timings.csv`, `filtered_timings.csv`, and `summary.csv`. Per-case JSON keeps
E2E samples in `timings_us`, generated-host NPU samples in `npu_timings_us`, and
the timing-version, NPU scope, and aggregation provenance. Raw and filtered CSV
rows retain `time_us` as E2E and add `npu_time_us` and the derived
`extra_time_us`. `summary.csv` reports raw and filtered mean, median, minimum,
maximum, and population standard deviation for E2E, NPU, and extra time.

The plotter writes `attention_e2e.png` under `plots/` or `plots/xdna2/`.
Override result roots with `--output-dir`; the plotter additionally accepts
`--summary` and its own `--output-dir`.

The plot is one grouped, stacked bar chart on a linear millisecond axis. Each
sequence length has an unfused baseline bar and a fused FlashAttention bar. The
dark lower segment is filtered mean NPU time, while the light upper segment is
filtered mean extra time. Both segments use shades of the implementation's
color, and the stack top is filtered mean E2E time. The legend contains only
`Unfused` and `Fused`; the infeasible marker is intentionally omitted. The
plotter requires both implementations at all six lengths, validates device,
mapping, and timing provenance, and rejects rows without paired component
means. Baseline/2048 has no timing, so a red cross replaces that bar.

### Timing and validation

Each runnable case performs one check against a stable float32 reference using
the exact bf16 inputs and `atol=5e-2`. A baseline comparison failure is recorded
with `status: "failed"`, `validation: "failed"`, and
`timed_validation_failure: true`, then still runs all requested warmups and
timed samples. Those complete samples participate in filtering and plots.
FlashAttention validation failures remain hard failures and are not benchmarked.
Baseline/2048 is recorded as `status: "infeasible"` with no validation or
timing. Defaults are 20 unrecorded warmups and 100 recorded complete calls.

E2E timing uses Python `time.perf_counter_ns()`. Compilation, input generation,
output allocation, and reference computation are outside this timer. For the
baseline it starts immediately before score GEMM and ends after value GEMM
returns; for FlashAttention it surrounds the complete fused module call. Thus
E2E includes module host-process launch, file and buffer transfers, xclbin
loading, XRT device/context and buffer creation, NPU execution and waits, output
synchronization, and copying results back to NumPy. The baseline intentionally
pays these costs for all four module calls.

NPU timing is parsed without backend changes from each generated `test.cpp` line
`NPU execution time: <value>us`. In that host program, the interval starts at
`auto start = ...` immediately before `kernel(...)` and ends at `auto end = ...`
immediately after `run.wait()`. It therefore excludes setup and host-to-device
synchronization before the launch and device-to-host synchronization after the
wait. A fused attention uses its one kernel interval. An unfused attention sums
the four ordered score, scale, softmax, and value-GEMM intervals.

The runner requires exactly
`(1 validation + warmups + iterations) * kernel_count` generated-host values,
discards the validation and warmup values, and pairs the remaining values with
the Python E2E samples. Every NPU value must be positive and no greater than its
paired E2E value; missing, malformed, excess, or inconsistent output fails the
case. `extra_time_us` is derived per sample as `E2E - NPU`, so it contains all
non-NPU work inside the E2E boundary.

The inclusive 1.5-IQR Tukey mask is calculated only from E2E samples and applied
to the paired E2E, NPU, and extra components together. Consequently, filtered
means preserve `E2E = NPU + extra`. Legacy JSON and CSV data remain readable by
the processor with empty component fields, but the timing-version signature
prevents resume and the plotter rejects legacy summaries until the complete
sweep has been regenerated.

### Attention flags

| Flag | Commands | Meaning |
| --- | --- | --- |
| `--device {xdna1,xdna2}` | `run`, `list`, `process`, plotter | Select the device and its default paths; defaults to `xdna1`. |
| `--implementation {baseline,flash,both}` | `run`, `list` | Select the four-module baseline, fused FlashAttention, or both; defaults to `both`. |
| `--seq-len SIZE [SIZE ...]` | `run`, `list` | Select from `64 128 256 512 1024 2048`; defaults to all six. Baseline/2048 produces an infeasible record without execution. |
| `--warmup COUNT` | `run` | Complete unrecorded calls per case; defaults to `20`. |
| `--iterations COUNT` | `run` | Complete recorded calls per case; defaults to `100`. |
| `--output-dir PATH` | `run`, `process`, plotter | Override the result or plot directory. |
| `--rerun` | `run` | Replace matching completed records. |
| `--keep-builds` | `run` | Retain per-case Allo projects and artifacts. |
| `--fail-fast` | `run` | Stop after the first hard failure. Timed baseline validation failures continue. |
| `--dry-run` | `run` | Print commands without checks, compilation, hardware access, or writes. |
| `--summary PATH` | plotter | Override the device-specific summary CSV. |

## Partial-NPU bf16 GEMM experiment

[`gemm_partial_npu.py`](gemm_partial_npu.py) compares three logical variants.
Plot names retain the columns-by-rows convention (`1x4` through `8x4`):

| Variant | Mapping primitives | Physical device width |
| --- | --- | --- |
| `manual` (Manual Template) | mlir-aie `n_aie_cols=W` | `W` |
| `compiled` (Compiled) | Allo `4xW`, except `W=1` uses `row_num=2,col_num=2` | `W`; the special 2x2 mapping still targets one column |
| `compiled-full-io` (Compiled (Full I/O)) | Allo `row_num=4,col_num=W` | all 4 XDNA1 columns or all 8 XDNA2 columns |

XDNA1 defaults to widths `1,2,4` and all four M/N/K sizes. Its 64 matrix
shapes produce 512 physical runs and 576 logical plot points. XDNA2 defaults to
widths `1,2,4,8`, with `N=512,1024,2048` and all four M/K sizes. Its 48
shapes produce 528 physical runs and 576 logical points. At the selected
device's full width (4 or 8), Compiled and Compiled (Full I/O) have the same
mapping and device, so one canonical `compiled` execution supplies both series.

Unsupported and non-divisible selections are rejected before execution. For
example, XDNA1 does not accept width 8, and XDNA2 width 8 does not accept
`N=256` with `n=64`.

Inspect either expansion without hardware:

```bash
python3 aie-experiments/gemm_partial_npu.py list
python3 aie-experiments/gemm_partial_npu.py list --device xdna2
```

Run and process the complete XDNA2 experiment:

```bash
python3 aie-experiments/gemm_partial_npu.py run \
  --device xdna2 \
  --mlir-aie-root /home/sf668/usr/mlir-aie
python3 aie-experiments/gemm_partial_npu.py process --device xdna2
```

The runner also supports `--variant`, `--columns`, `--M`, `--N`, `--K`,
`--warmup`, `--iterations`, `--output-dir`, `--rerun`, `--keep-builds`,
`--fail-fast`, and `--dry-run`. Preview the special Compiled 1x4 mapping on
XDNA2 with:

```bash
python3 aie-experiments/gemm_partial_npu.py run \
  --device xdna2 --variant compiled --columns 1 \
  --M 256 --N 512 --K 256 --dry-run
```

The printed Allo worker command includes `--columns 2 --rows 2` and
`--device-columns 1`. Manual remains `n_aie_cols=1`; Compiled Full I/O remains
`row_num=4,col_num=1` and uses the selected device's full physical width.

All Allo partial cases automatically continue timing after a bf16 output
comparison failure. Their JSON/CSV rows retain failed validation status, the
traceback, complete samples, and `timed_validation_failure: true`; they remain
filterable and resumable without causing a nonzero sweep result. Build,
runtime, device, and timing failures remain hard failures. Every work directory
uses the same aggressive cleanup policy described above.

Generate the device-specific plots with:

```bash
python3 aie-experiments/plot_partial_npu.py
python3 aie-experiments/plot_partial_npu.py --device xdna2
```

XDNA1 writes `gemm_bf16_1x4.png`, `gemm_bf16_2x4.png`, and
`gemm_bf16_4x4.png` under `plots/`. XDNA2 adds `gemm_bf16_8x4.png` and
writes all four under `plots/xdna2/`. Each plot contains all three logical
series and Tukey-filtered min/max envelopes. Completeness and device provenance
are validated independently per width; an incomplete or mismatched width is
skipped without blocking the others.
