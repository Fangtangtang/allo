# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLIR-AIR backend helpers used by Allo tests.

The public entry used in tests is :func:`_call_prj`.

Design goals for `_call_prj` (tests/dataflow/air/*_from_project.py):
  * Never emit compilation artifacts outside the provided `project` directory.
  * Support external AIE kernels referenced by AIR MLIR via `link_with = "..."`.

We rely on the upstream mlir-air python bindings for running on XRT.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import filelock

from air.backend.xrt import XRTBackend, XRTCompileArtifact
from air.ir import Context, Location, Module


@contextlib.contextmanager
def _pushd(new_dir: str | os.PathLike):
    """Temporarily change working directory."""
    prev = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev)


def _repo_root_from_this_file() -> Path:
    # allo/backend/air.py -> <repo_root>/allo/backend/air.py; repo_root is 3 levels up.
    return Path(__file__).resolve().parents[2]


def _resolve_project_dir(project: str) -> Path:
    """Resolve `project` to an existing directory.

    Tests pass relative names like "vadd_prj" or "external_prj".
    Depending on how the test is launched, the CWD may vary.

    We attempt:
      1) `project` as given (relative to CWD)
      2) relative to <repo_root>/tests/dataflow/air
    """
    p = Path(project)
    if p.is_dir():
        return p.resolve()

    candidate = _repo_root_from_this_file() / "tests" / "dataflow" / "air" / project
    if candidate.is_dir():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Project directory '{project}' not found. Tried '{p.resolve()}' and '{candidate}'."
    )


def _detect_target_device() -> str:
    """Best-effort detection of the NPU model for aircc."""
    target_device = "npu2"
    xrtsmi = "/opt/xilinx/xrt/bin/xrt-smi"
    try:
        if not os.path.exists(xrtsmi):
            return target_device
        result = subprocess.run(
            [xrtsmi, "examine"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        out = result.stdout.decode("utf-8", errors="ignore").splitlines()
        p = re.compile(r"[\|]?(\[.+:.+:.+\]).+\|(RyzenAI-(npu\d)|NPU (\w+))\W*\|")
        for l in out:
            m = p.match(l)
            if not m:
                continue
            model = "unknown"
            if m.group(3):
                model = str(m.group(3))
            if m.group(4):
                model = str(m.group(4))
            if model in ["npu1", "Phoenix"]:
                target_device = "npu1"
            elif model in ["npu4", "Strix"]:
                target_device = "npu2"
            break
    except Exception:
        pass

    return target_device


def _infer_bit_width_from_mlir(mlir_text: str) -> int:
    # Tiny heuristic sufficient for the unit tests.
    if re.search(r"(ui8|i8)\b", mlir_text):
        return 8
    if re.search(r"(ui16|i16|f16|bf16)\b", mlir_text):
        return 16
    return 32


def _extract_link_with_objects(mlir_text: str) -> list[str]:
    # Example: attributes {link_with = "passThrough.cc.o"}
    return re.findall(r'link_with\s*=\s*"([^"]+)"', mlir_text)


def _maybe_compile_external_kernel_objects(
    *,
    project_dir: Path,
    build_dir: Path,
    mlir_text: str,
):
    """Compile external kernel source files referenced by `link_with`.

    For the tests, top.mlir references `passThrough.cc.o` and the source
    `passThrough.cc` lives under the project directory.

    We compile into <project>/build to keep artifacts inside the project.
    """

    obj_names = _extract_link_with_objects(mlir_text)
    if not obj_names:
        return

    peano = os.environ.get("PEANO_INSTALL_DIR", "")
    clangpp = os.path.join(peano, "bin", "clang++") if peano else ""
    if not (clangpp and os.path.isfile(clangpp)):
        raise RuntimeError(
            "External kernels detected (link_with=...), but PEANO_INSTALL_DIR/bin/clang++ was not found."
        )

    aie_opt = shutil.which("aie-opt")
    if not aie_opt:
        raise RuntimeError(
            "External kernels detected, but 'aie-opt' was not found in PATH."
        )
    aieopt_dir = str(Path(aie_opt).resolve().parent.parent)

    target_device = _detect_target_device()
    aie_target = "aie2p" if target_device == "npu2" else "aie2"
    bit_width = _infer_bit_width_from_mlir(mlir_text)

    warning_flags = [
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
    ]

    for obj in obj_names:
        out_obj = build_dir / obj
        if out_obj.exists():
            continue

        # Infer source file from the object file name.
        src_candidates: list[Path] = []
        if obj.endswith(".cc.o"):
            src_candidates.append(project_dir / obj[: -len(".o")])
        elif obj.endswith(".cpp.o"):
            src_candidates.append(project_dir / obj[: -len(".o")])
        elif obj.endswith(".c.o"):
            src_candidates.append(project_dir / obj[: -len(".o")])
        elif obj.endswith(".o"):
            stem = obj[:-2]
            src_candidates.extend(
                [
                    project_dir / f"{stem}.cc",
                    project_dir / f"{stem}.cpp",
                    project_dir / f"{stem}.c",
                    project_dir / stem,
                ]
            )
        else:
            src_candidates.append(project_dir / obj)

        src = next((p for p in src_candidates if p.exists()), None)
        if src is None:
            raise FileNotFoundError(
                f"MLIR requests link_with='{obj}' but no source found. Tried: {', '.join(str(p) for p in src_candidates)}"
            )

        cmd = [
            clangpp,
            "-O2",
            "-std=c++20",
            f"--target={aie_target}-none-unknown-elf",
            *warning_flags,
            "-DNDEBUG",
            f"-I{aieopt_dir}/include",
            f"-DBIT_WIDTH={bit_width}",
            "-c",
            str(src),
            "-o",
            str(out_obj),
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to compile external kernel object.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Output:\n{proc.stdout.decode('utf-8', errors='ignore')}"
            )


def _compile_air_module_to_project_build(
    *,
    mlir_module: Module,
    build_dir: Path,
    xclbin_name: str = "air.xclbin",
    insts_name: str = "air.insts.bin",
    kernel: str = "MLIR_AIE",
) -> XRTCompileArtifact:
    """Compile AIR module using aircc but force all outputs under build_dir.

    We cannot rely on XRTBackend.compile() directly because it does not expose
    aircc's --tmpdir, and the test environment blocks relative-path writes.

    This function also mirrors XRTBackend.compile's peano/xchesscc selection.
    """

    import air.compiler.aircc.main as aircc

    target_device = _detect_target_device()

    # Absolute paths to satisfy sandbox rules.
    tmpdir = str((build_dir / "air_project").resolve())
    xclbin_path = str((build_dir / xclbin_name).resolve())
    insts_path = str((build_dir / insts_name).resolve())

    # aircc.run uses this only for naming intermediate files.
    air_mlir_file = str((build_dir / "air.mlir").resolve())

    aircc_args = [
        "--device",
        target_device,
        "--tmpdir",
        tmpdir,
        air_mlir_file,
        "-o",
        xclbin_path,
        "-i",
        insts_path,
    ]

    # Ensure peano toolchain is used when available (matches XRTBackend.compile).
    peano_install_dir = os.environ.get("PEANO_INSTALL_DIR", "")
    if peano_install_dir and os.path.isdir(peano_install_dir):
        aircc_args += ["--peano", peano_install_dir, "--no-xchesscc", "--no-xbridge"]
    else:
        aircc_args += ["--xchesscc", "--xbridge"]

    aircc.run(mlir_module, aircc_args)

    return XRTCompileArtifact(xclbin=xclbin_path, kernel=kernel, insts=insts_path)


def _call_prj(project: str, output_idx: list[int], *args):
    """Compile and run an AIR project directory.

    Parameters
    ----------
    project:
        Path or name of a directory containing `top.mlir` and optionally
        external kernel sources.
    output_idx:
        Indices in `args` that correspond to output buffers.
    args:
        Numpy arrays, matching the compiled module interface.

    Guarantees
    ----------
    All compilation artifacts are emitted under <project>/build.
    """

    project_dir = _resolve_project_dir(project)
    top_mlir_path = project_dir / "top.mlir"
    if not top_mlir_path.exists():
        raise FileNotFoundError(f"Missing '{top_mlir_path}'.")

    build_dir = project_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    mlir_text = top_mlir_path.read_text(encoding="utf-8")

    # External kernels (if any) must be compiled into build_dir so that aiecc
    # can find them when linking.
    _maybe_compile_external_kernel_objects(
        project_dir=project_dir, build_dir=build_dir, mlir_text=mlir_text
    )

    with Context() as _ctx, Location.unknown():
        mlir_module = Module.parse(mlir_text)

    # Compile (forces outputs into build_dir).
    with _pushd(build_dir):
        artifact = _compile_air_module_to_project_build(
            mlir_module=mlir_module, build_dir=build_dir
        )

    # Execute.
    backend = XRTBackend()
    with filelock.FileLock("/tmp/npu.lock"):
        module_function = backend.load(artifact)
        actual_outputs = module_function(*args)
    backend.unload()

    # Copy back outputs.
    for idx in output_idx:
        try:
            args[idx][...] = actual_outputs[idx]
        except Exception:
            args[idx][:] = actual_outputs[idx]
