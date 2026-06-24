#!/usr/bin/env python3
"""Stage 0.1 remote H100 output-preservation runner for SpatialKDV."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


EXACT_P1_ROOT = Path("/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/GPU-kernel-density-exact-stage0-p1")
sys.path.insert(0, str(EXACT_P1_ROOT / "scripts"))

from stage0p1_output_utils import Attempt, run_attempts, run_env_with_cuda


SPATIAL_ROOT = Path("/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/baselines/GPUParallelSpatialAdaptiveKDE")
BUILD_DIR = SPATIAL_ROOT / "build-stage0-p1-remote-h100"
BINARY = BUILD_DIR / "kde_cuda_kdtr"
CUDA_HOME = Path("/usr/local/cuda-12.4")
RUN_ROOT = Path(
    os.environ.get(
        "STAGE0P1_RUN_ROOT",
        "/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/tmp-results/stage0-p1/remote-h100",
    )
)
DATA_ROOT = Path("/home/lxheq/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation")
MACHINE = "remote-h100"
METHOD = "SpatialKDV"
METHOD_TOKEN = "spatialkdv"
PRECISION = "FP32"
SCALE = "scott_diag b=1"
SCALE_VALUE = "1"
TIMEOUT_SECONDS = 3600
CONFIGURE_TIMEOUT_SECONDS = 600
BUILD_TIMEOUT_SECONDS = 1800
CUDA_ARCH = "90"
BUILD_TYPE = "Release"

SUPPORTED_WORKLOADS = (
    ("kdv_home_visual", DATA_ROOT / "home_visualization/HT_Sensor_dataset_vis_X.data", 1920, 2560, 4915200),
    ("kdv_susy_visual", DATA_ROOT / "susy_visualization/SUSY_vis_X.data", 1920, 2560, 4915200),
)
UNSUPPORTED_WORKLOADS = ("svm_susy", "svm_home", "svm_miniboone")


def run_setup(command: list[str], timeout_seconds: int) -> None:
    completed = subprocess.run(
        command,
        cwd=SPATIAL_ROOT,
        env=run_env_with_cuda(CUDA_HOME),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{completed.stdout}")


def build_binary() -> None:
    run_setup(
        [
            "cmake",
            "-S",
            str(SPATIAL_ROOT / "kde_cuda"),
            "-B",
            str(BUILD_DIR),
            f"-DCMAKE_BUILD_TYPE={BUILD_TYPE}",
            f"-DCUDA_DEVICE_VERSION={CUDA_ARCH}",
        ],
        CONFIGURE_TIMEOUT_SECONDS,
    )
    run_setup(["cmake", "--build", str(BUILD_DIR), "--target", "kde_cuda_kdtr", "-j", str(os.cpu_count() or 1)], BUILD_TIMEOUT_SECONDS)


def output_rel(workload: str, rows: int, cols: int) -> Path:
    name = f"{MACHINE}_spatialkdv_fp32_{workload}_{rows}x{cols}_scott_diag_b1.out"
    return Path("outputs") / METHOD_TOKEN / "fp32" / name


def planned_attempts() -> list[Attempt]:
    attempts: list[Attempt] = []
    for workload in UNSUPPORTED_WORKLOADS:
        attempts.append(
            Attempt(
                machine=MACHINE,
                method=METHOD,
                precision=PRECISION,
                workload=workload,
                scale=SCALE,
                command=None,
                cwd=SPATIAL_ROOT,
                output_rel=None,
                log_rel=None,
                timeout_seconds=0,
                status="unsupported_workload",
            )
        )
    env = run_env_with_cuda(CUDA_HOME)
    for workload, data_path, rows, cols, expected in SUPPORTED_WORKLOADS:
        rel = output_rel(workload, rows, cols)
        attempts.append(
            Attempt(
                machine=MACHINE,
                method=METHOD,
                precision=PRECISION,
                workload=workload,
                scale=SCALE,
                command=(
                    str(BINARY),
                    str(data_path),
                    str(rows),
                    str(cols),
                    "--scott-diag",
                    SCALE_VALUE,
                    str(RUN_ROOT / rel),
                ),
                cwd=SPATIAL_ROOT,
                output_rel=rel,
                log_rel=Path("logs") / METHOD_TOKEN / f"{rel.stem}.log",
                timeout_seconds=TIMEOUT_SECONDS,
                expected_value_count=expected,
                env=env,
            )
        )
    return attempts


def main() -> int:
    build_binary()
    fragment = RUN_ROOT / "fragments" / f"{METHOD_TOKEN}.csv"
    run_attempts(planned_attempts(), RUN_ROOT, fragment)
    print(f"[STAGE0P1] manifest_fragment={fragment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
