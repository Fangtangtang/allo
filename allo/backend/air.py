# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal mlir-air project runner for backend tests.

The unit tests in `tests/dataflow/air/*_from_project.py` provide a pre-generated
AIR dialect `top.mlir` under a project directory (e.g. `vadd.prj`). This module
compiles that IR with mlir-air's `aircc` via `air.backend.xrt.XRTBackend`,
optionally builds any external AIE kernels referenced via `link_with`, and
executes the result through XRT.

Design goals:
  * Keep all intermediate/compiled artifacts under the given project dir.
  * Be robust to repeated calls by using file locks.
  * Keep implementation simple (only supports the needs of the unit tests).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import filelock

from air.backend.xrt import XRTBackend, XRTCompileArtifact
from air.ir import Context, Location, Module


def _resolve_project_dir(project: str) -> Path:
    """Resolve a project directory argument used by tests.

    Tests pass e.g. "vadd.prj" and expect it to resolve to
    `<allo repo>/tests/dataflow/air/vadd.prj`.
    """

    p = Path(project)
    if p.is_absolute() and p.exists():
        return p

    # 1) Try relative to current working directory.
    cand = (Path.cwd() / p).resolve()
    if cand.exists():
        return cand

    # 2) Try relative to this repo's tests/dataflow/air directory.
    # air.py is at <repo>/allo/backend/air.py
    repo_root = Path(__file__).resolve().parents[2]
    cand = (repo_root / "tests" / "dataflow" / "air" / p).resolve()
    if cand.exists():
        return cand

    # 3) Try relative to repo root.
    cand = (repo_root / p).resolve()
    if cand.exists():
        return cand

    # Fall back to absolute resolution for error message.
    return (Path.cwd() / p).resolve()


def _detect_aie_target() -> str:
    """Return clang AIE target ("aie2" or "aie2p").

    Falls back to "aie2" if detection fails.
    """

    try:
        xrtsmi = Path("/opt/xilinx/xrt/bin/xrt-smi")
        if not xrtsmi.exists():
            return "aie2"
        res = subprocess.run(
            [str(xrtsmi), "examine"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        pat = re.compile(r"[\|]?(\[.+:.+:.+\]).+\|(RyzenAI-(npu\d)|NPU (\w+))\W*\|")
        for line in res.stdout.splitlines():
            m = pat.match(line)
            if not m:
                continue
            model = m.group(3) or m.group(4) or "unknown"
            if model in ("npu1", "Phoenix"):
                return "aie2"
            if model in ("npu4", "Strix"):
                return "aie2p"
            break
    except Exception:
        pass
    return "aie2"


def _find_link_with_objects(mlir_text: str) -> set[str]:
    """Extract `link_with = "..."` object names from MLIR."""

    return set(re.findall(r'link_with\s*=\s*"([^"]+)"', mlir_text))


def _maybe_compile_external_object(project_dir: Path, obj_name: str) -> None:
    """Compile an external kernel object file if it is missing.

    Minimal support needed by the external-kernel unit test.
    """

    obj_path = project_dir / obj_name
    if obj_path.exists():
        return

    # Heuristic: map "foo.cc.o" -> "foo.cc"; "foo.o" -> "foo.cc" etc.
    base = obj_name
    if base.endswith(".o"):
        base = base[: -len(".o")]

    candidates: list[Path] = []
    for ext in ("", ".cc", ".cpp", ".cxx"):
        cand = project_dir / f"{base}{ext}"
        if cand.exists() and cand.is_file():
            candidates.append(cand)

    if not candidates and base.endswith((".cc", ".cpp", ".cxx")):
        cand = project_dir / base
        if cand.exists() and cand.is_file():
            candidates.append(cand)

    if not candidates:
        raise RuntimeError(
            f"External object '{obj_name}' referenced by MLIR but not found in project, "
            f"and no matching source could be inferred under {project_dir}."
        )

    src_path = candidates[0]

    peano = os.environ.get("PEANO_INSTALL_DIR", "")
    if not peano:
        raise RuntimeError(
            "PEANO_INSTALL_DIR is not set; cannot compile external AIE kernels."
        )

    clangpp = Path(peano) / "bin" / "clang++"
    if not clangpp.exists():
        raise RuntimeError(f"Cannot find PEANO clang++ at '{clangpp}'.")

    aie_target = _detect_aie_target()
    triple = f"--target={aie_target}-none-unknown-elf"

    cmd = [
        str(clangpp),
        "-O2",
        "-std=c++20",
        triple,
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-DNDEBUG",
        "-I",
        os.path.join(os.environ.get("MLIR_AIE_INSTALL_DIR", ""), "include"),
        "-I",
        os.environ.get("MLIR_AIE_EXTERNAL_KERNEL_DIR", ""),
        "-I.",
        "-c",
        str(src_path.name),
        "-o",
        str(obj_path.name),
    ]

    proc = subprocess.run(cmd, cwd=str(project_dir), check=False)
    if proc.returncode != 0 or not obj_path.exists():
        raise RuntimeError(
            f"Failed to compile external kernel '{src_path.name}' to '{obj_path.name}'.\n"
            f"Command: {' '.join(cmd)}"
        )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _call_prj(project: str, output_idx: list[int], *args):
    """Compile `top.mlir` under `project` and execute it on XRT.

    * `args` is a mixed list of numpy input and output arrays.
    * `output_idx` gives which entries in `args` must be updated in-place.

    All intermediate files are kept under the `project` directory.
    """

    project_dir = _resolve_project_dir(project)
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    top_mlir = project_dir / "top.mlir"
    if not top_mlir.exists():
        raise FileNotFoundError(f"Expected MLIR file at: {top_mlir}")

    # Locks: compilation lock and device-use lock, both within project.
    compile_lock_path = project_dir / ".allo_air.lock"
    npu_lock_path = project_dir / ".npu.lock"

    build_dir = project_dir / "build"
    _ensure_dir(build_dir)

    xclbin_path = build_dir / "final.xclbin"
    insts_path = build_dir / "air.insts.bin"

    mlir_text = top_mlir.read_text(encoding="utf-8")
    linked_objs = _find_link_with_objects(mlir_text)

    # Compile under project-specific lock to avoid races.
    with filelock.FileLock(str(compile_lock_path)):
        for obj in linked_objs:
            _maybe_compile_external_object(project_dir, obj)

        def _mtime(p: Path) -> float:
            return p.stat().st_mtime if p.exists() else -1.0

        newest_dep = _mtime(top_mlir)
        for obj in linked_objs:
            newest_dep = max(newest_dep, _mtime(project_dir / obj))

        need_rebuild = (
            (not xclbin_path.exists())
            or (not insts_path.exists())
            or (_mtime(xclbin_path) < newest_dep)
            or (_mtime(insts_path) < newest_dep)
        )

        if need_rebuild:
            with Context() as ctx, Location.unknown(ctx):
                ctx.allow_unregistered_dialects = True
                air_mod = Module.parse(mlir_text)
                backend = XRTBackend(verbose=False)
                old_cwd = os.getcwd()
                try:
                    os.chdir(str(project_dir))
                    backend.compile(
                        air_mod,
                        xclbin=str(xclbin_path),
                        kernel="MLIR_AIE",
                        insts=str(insts_path),
                    )
                finally:
                    os.chdir(old_cwd)
                    backend.unload()

    # Execute under a device lock.
    with filelock.FileLock(str(npu_lock_path)):
        backend = XRTBackend(verbose=False)
        old_cwd = os.getcwd()
        try:
            os.chdir(str(project_dir))
            fn = backend.load(
                XRTCompileArtifact(
                    xclbin=str(xclbin_path),
                    kernel="MLIR_AIE",
                    insts=str(insts_path),
                )
            )
            results = fn(*args)
        finally:
            os.chdir(old_cwd)
            backend.unload()

    # Write outputs back into provided output buffers.
    for idx in output_idx:
        if idx < 0 or idx >= len(args):
            raise IndexError(f"output_idx contains out-of-range index {idx}")
        out_arr = args[idx]
        res_arr = results[idx]
        try:
            res_arr = res_arr.reshape(out_arr.shape)
        except Exception:
            pass
        out_arr[...] = res_arr

    return None
