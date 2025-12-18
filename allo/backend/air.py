# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLIR-AIR backend helpers.

This module implements a minimal project-based runner used by unit tests.

Requirements satisfied:
- Read AIR dialect MLIR from `<project>/top.mlir`.
- Compile using mlir-air's compiler driver (aircc) and run via XRT.
- Keep *all* intermediate files under the given project directory.
- Support external kernels (`*.cc/*.cpp`) compiled with `$PEANO_INSTALL_DIR/bin/clang++`.

The unit tests call :func:`_call_prj` directly.

Additionally, this file provides :func:`convert` which translates an Allo-generated
MLIR module (stored in `<project>/allo.mlir`) into an AIR dialect module
(stored in `<project>/top.mlir`).

NOTE: The conversion implemented here is intentionally minimal: it targets the
vector tests in `tests/dataflow/air/`.
"""

from __future__ import annotations

import inspect
import os
import subprocess
from pathlib import Path

import filelock
import numpy as np

from air.backend.xrt import XRTBackend, XRTCompileArtifact
from air.ir import (
    Context,
    Location,
    Module,
)

# aircc driver (compiler)
import air.compiler.aircc.main as aircc


def _call_prj(project: str, output_idx: list[int], *args):
    def _resolve_project_dir(project: str) -> Path:
        """Resolve a project directory from a potentially relative name."""

        p = Path(project)
        if p.is_absolute():
            return p

        # 1) relative to CWD
        if p.exists():
            return p.resolve()

        # 2) relative to caller file directory
        for frame_info in inspect.stack()[1:]:
            f = frame_info.filename
            if not f:
                continue
            if Path(f).resolve() == Path(__file__).resolve():
                continue
            cand = (Path(f).resolve().parent / p).resolve()
            if cand.exists():
                return cand

        raise FileNotFoundError(
            f"AIR project directory '{project}' not found. Tried CWD='{Path.cwd()}', caller-relative, and repo-relative paths."
        )

    def _compile_external_kernel(src: Path, project_dir: Path) -> Path:
        """Compile a single external kernel source to an object file."""

        clangpp = Path(os.environ.get("PEANO_INSTALL_DIR")) / "bin" / "clang++"
        if not clangpp.exists():
            raise RuntimeError(f"Cannot find '{clangpp}'.")

        mlir_aie = os.environ.get("MLIR_AIE_INSTALL_DIR")
        if not mlir_aie:
            raise RuntimeError(
                "MLIR_AIE_INSTALL_DIR is not set (required for external kernel includes)."
            )
        target = "aie2p" if os.getenv("NPU2") == "1" else "aie2"
        cmd: list[str] = [
            str(clangpp),
            "-O2",
            "-std=c++20",
            f"--target={target}-none-unknown-elf",
            "-Wno-parentheses",
            "-Wno-attributes",
            "-Wno-macro-redefined",
            "-DNDEBUG",
            "-I",
            str(Path(mlir_aie) / "include"),
            "-I",
            str(project_dir),
            "-I",
            os.environ.get("MLIR_AIE_EXTERNAL_KERNEL_DIR"),
            "-c",
            str(src),
            "-o",
            str(project_dir / f"{src.name}.o"),
        ]

        proc = subprocess.run(cmd, cwd=str(project_dir), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to compile external kernel.\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )

    def _compile_air_project(
        air_module: Module,
        project_dir: Path,
        build_dir: Path,
        xclbin_path: Path,
        insts_path: Path,
    ) -> XRTCompileArtifact:
        """Compile AIR module to xclbin+insts, writing everything under project_dir."""
        # Absolute tmpdir to satisfy sandboxing (relative paths can be blocked by LD_PRELOAD).
        tmpdir = build_dir / "aircc_tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)

        aircc_options: list[str] = [
            "--device",
            "npu2" if os.getenv("NPU2") == "1" else "npu1",
            "air.mlir",  # only used for naming; the module is passed in-memory
            "--tmpdir",
            str(tmpdir),
            "-o",
            str(xclbin_path),
            "-i",
            str(insts_path),
            "--output-format",
            "xclbin",
            "--peano",
            os.environ.get("PEANO_INSTALL_DIR"),
            "--no-xchesscc",
            "--no-xbridge",
        ]

        # Run aircc from the *project* directory so relative `link_with = "*.o"`
        # resolves to the object files we compiled into project_dir.
        old_cwd = os.getcwd()
        try:
            os.chdir(str(project_dir))
            aircc.run(air_module, aircc_options)
        finally:
            os.chdir(old_cwd)

        return XRTCompileArtifact(str(xclbin_path), "MLIR_AIE", str(insts_path))

    project_dir = _resolve_project_dir(project)
    top_mlir = project_dir / "top.mlir"
    if not top_mlir.exists():
        raise FileNotFoundError(f"'{top_mlir}' does not exist")

    np_args: list[np.ndarray] = []
    for a in args:
        if not isinstance(a, np.ndarray):
            raise TypeError(f"AIR backend expects numpy.ndarray args, got {type(a)}")
        np_args.append(a)

    build_dir = project_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Locks live in the project directory (not /tmp).
    prj_lock_path = project_dir / ".allo_air_prj.lock"
    npu_lock_path = project_dir / ".allo_npu.lock"

    with filelock.FileLock(str(prj_lock_path)):
        # 1) Compile external kernels if any.
        srcs = sorted(project_dir.glob("*.cc"))
        srcs = [p for p in srcs if p.is_file() and p.parent == project_dir]
        for src in srcs:
            _compile_external_kernel(src, project_dir)

        # 2) Parse MLIR.
        mlir_txt = top_mlir.read_text(encoding="utf-8")
        with Context() as ctx, Location.unknown():
            air_module = Module.parse(mlir_txt, ctx)

        # 3) Compile AIR->xclbin/insts (all artifacts under project_dir).
        artifact = _compile_air_project(
            air_module,
            project_dir=project_dir,
            build_dir=build_dir,
            xclbin_path=build_dir / "final.xclbin",
            insts_path=build_dir / "insts.bin",
        )

        # 4) Execute using XRT.
        backend = XRTBackend(verbose=False)
        with filelock.FileLock(str(npu_lock_path)):
            invoker = backend.load(artifact)
            results = invoker(*np_args)
        backend.unload()

    # 5) Copy results back.
    for idx in output_idx:
        out = np_args[idx]
        got = np.asarray(results[idx]).reshape(out.shape)
        out[...] = got
