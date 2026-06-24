#!/usr/bin/env python3
"""Stage 0 local 3080Ti SpatialKDV runner.

The script writes temporary evidence only. Accepted compact evidence is promoted
to persistent storage manually after review.
"""

from __future__ import annotations

import csv
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


SPATIAL_ROOT = Path("/home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/baselines/GPUParallelSpatialAdaptiveKDE")
BUILD_DIR = SPATIAL_ROOT / "build-stage0-local-3080ti"
BINARY = BUILD_DIR / "kde_cuda_kdtr"
CUDA_HOME = Path("/usr/local/cuda-12.6")

DATA_ROOT = Path("/home/ubuntu/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation")
TMP_RESULT_ROOT = Path("/home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/tmp-results/stage0/spatialkdv")
PERSISTENT_RESULT_ROOT = Path("/home/ubuntu/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation/exact/experiments/stage0")
EXACT_REPO_ROOT = Path("/home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/GPU-kernel-density-exact")
BASIC_GROUND_TRUTH_ROOT = PERSISTENT_RESULT_ROOT / "ground_truth"

MACHINE = "local-3080ti"
METHOD = "SpatialKDV"
METHOD_TOKEN = "spatialkdv"
ENGINE = "kde_cuda_kdtr_exact_kdv"
BACKEND = "GPU"
PRECISION = "FP32"
CUDA_ARCH = "86"
BUILD_TYPE = "Release"
EXPECTED_TIMING_SCOPE = "in_memory_spatial_kdv_kernel"
KDV_SCOTT_B = "1"
SMOKE_SCOTT_B = "1"
KDV_ROWS = 1920
KDV_COLS = 2560
SMOKE_KDV_ROWS = 256
SMOKE_KDV_COLS = 256
BUILD_JOBS = os.cpu_count() or 1
CONFIGURE_TIMEOUT_SECONDS = 600
BUILD_TIMEOUT_SECONDS = 1800
SMOKE_TIMEOUT_SECONDS = 600
KDV_TIMEOUT_SECONDS = 3600
FP32_ABS_TOLERANCE = 1e-2
FP32_REL_TOLERANCE = 1e-4
GPU_QUERY_COMMAND = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version,compute_cap",
    "--format=csv,noheader",
]

HOME_VIS_X = DATA_ROOT / "home_visualization/HT_Sensor_dataset_vis_X.data"
SUSY_VIS_X = DATA_ROOT / "susy_visualization/SUSY_vis_X.data"

SMOKE_SUSY_VIS_X = EXACT_REPO_ROOT / "data/SUSY_vis_10000_X.data"
SMOKE_KDV_REF = EXACT_REPO_ROOT / "results/SUSY_vis_10000_kdv_256x256_basic_scan_fp64_scott_diag_b1.out"


@dataclass(frozen=True)
class Workload:
    run_group: str
    workload: str
    dataset: str
    data_path: Path
    grid_rows: int
    grid_cols: int
    scale_mode: str
    scale_value: str


@dataclass
class RunRecord:
    workload: Workload
    output_path: Path
    log_path: Path
    command: list[str]
    return_code: str
    timeout_seconds: int
    cli_wall_seconds: float
    timing_scope: str
    execution_seconds: str
    query_count_from_binary: str
    qps: str
    status: str


def run_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_HOME"] = str(CUDA_HOME)
    env["PATH"] = f"{CUDA_HOME / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{CUDA_HOME / 'lib64'}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    return env


def shell_join(command: Iterable[str]) -> str:
    return shlex.join([str(part) for part in command])


def run_shell_capture(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=SPATIAL_ROOT,
        env=run_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.rstrip()


def matrix_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().split()
    if len(first) < 2:
        raise ValueError(f"Invalid matrix header: {path}")
    return int(first[0]), int(first[1])


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def parse_metrics(log_text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for key in ("timing_scope", "execution_seconds", "query_count", "qps", "precision"):
        match = re.search(rf"^{key}:\s*(.+?)\s*$", log_text, re.MULTILINE)
        if match:
            metrics[key] = match.group(1)
    return metrics


def iter_floats(path: Path) -> Iterable[float]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            for token in line.split():
                yield float(token)


def compare_numeric_files(output_path: Path, reference_path: Path) -> dict[str, str]:
    max_abs = 0.0
    max_rel = 0.0
    failure_count = 0
    count = 0
    output_iter = iter_floats(output_path)
    reference_iter = iter_floats(reference_path)

    while True:
        sentinel = object()
        output_value = next(output_iter, sentinel)
        reference_value = next(reference_iter, sentinel)
        if output_value is sentinel and reference_value is sentinel:
            break
        if output_value is sentinel or reference_value is sentinel:
            failure_count += 1
            break

        count += 1
        diff = abs(float(output_value) - float(reference_value))
        max_abs = max(max_abs, diff)
        if reference_value != 0.0:
            max_rel = max(max_rel, diff / abs(float(reference_value)))
        threshold = max(FP32_ABS_TOLERANCE, FP32_REL_TOLERANCE * abs(float(reference_value)))
        if diff > threshold:
            failure_count += 1

    return {
        "value_count": str(count),
        "max_abs_err": f"{max_abs:.12g}",
        "max_rel_err": f"{max_rel:.12g}",
        "failure_count": str(failure_count),
        "tolerance": f"abs={FP32_ABS_TOLERANCE:g};rel={FP32_REL_TOLERANCE:g}",
        "status": "ok" if failure_count == 0 else "failed_tolerance",
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_command(commands_file: Path, command: list[str], timeout_seconds: int | None = None) -> None:
    with commands_file.open("a", encoding="utf-8") as handle:
        handle.write(f"cd {shlex.quote(str(SPATIAL_ROOT))}\n")
        if timeout_seconds is not None:
            handle.write(f"# timeout_seconds: {timeout_seconds}\n")
        handle.write(shell_join(command))
        handle.write("\n\n")


def timeout_output_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_command(
    command: list[str],
    log_path: Path,
    commands_file: Path,
    timeout_seconds: int,
) -> tuple[str, float, str, bool]:
    append_command(commands_file, command, timeout_seconds)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=SPATIAL_ROOT,
            env=run_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        wall_seconds = time.perf_counter() - start
        partial_output = timeout_output_to_text(exc.stdout or exc.output)
        log_text = (
            partial_output
            + f"\n[TIMEOUT] Command exceeded timeout_seconds={timeout_seconds}.\n"
            + f"[TIMEOUT] cli_wall_seconds={wall_seconds:.9f}\n"
        )
        write_text(log_path, log_text)
        return "timeout", wall_seconds, log_text, True

    wall_seconds = time.perf_counter() - start
    write_text(log_path, completed.stdout)
    return str(completed.returncode), wall_seconds, completed.stdout, False


def run_inventory_command(label: str, command: list[str]) -> str:
    output = run_shell_capture(command)
    return f"{label}:\n{output}\n"


def collect_gpu_summary() -> dict[str, str]:
    output = run_shell_capture(GPU_QUERY_COMMAND)
    first_line = output.splitlines()[0] if output else ""
    parts = [part.strip() for part in first_line.split(",")]
    while len(parts) < 4:
        parts.append("")
    return {
        "gpu_name": parts[0],
        "gpu_memory_total": parts[1],
        "gpu_driver_version": parts[2],
        "gpu_compute_cap": parts[3],
    }


def write_inventory(path: Path, timestamp: str, git_branch: str, git_commit: str) -> None:
    content = [
        f"timestamp: {timestamp}",
        f"machine: {MACHINE}",
        f"hostname: {socket.gethostname()}",
        f"spatial_root: {SPATIAL_ROOT}",
        f"build_dir: {BUILD_DIR}",
        f"binary: {BINARY}",
        f"data_root: {DATA_ROOT}",
        f"tmp_result_root: {TMP_RESULT_ROOT}",
        f"persistent_result_root: {PERSISTENT_RESULT_ROOT}",
        f"basic_ground_truth_root: {BASIC_GROUND_TRUTH_ROOT}",
        f"method: {METHOD}",
        f"engine: {ENGINE}",
        f"backend: {BACKEND}",
        f"precision: {PRECISION}",
        f"cuda_arch: {CUDA_ARCH}",
        f"build_type: {BUILD_TYPE}",
        f"timing_scope: {EXPECTED_TIMING_SCOPE}",
        f"kdv_scott_b: {KDV_SCOTT_B}",
        f"git_branch: {git_branch}",
        f"git_commit: {git_commit}",
        "",
        run_inventory_command("git_status", ["git", "status", "--short"]),
        run_inventory_command("uname", ["uname", "-a"]),
        run_inventory_command("lscpu", ["bash", "-lc", "lscpu | sed -n '1,40p'"]),
        run_inventory_command("nvidia-smi", GPU_QUERY_COMMAND),
        run_inventory_command("nvcc", ["nvcc", "--version"]),
        run_inventory_command("gcc", ["bash", "-lc", "gcc --version | head -1"]),
        run_inventory_command("g++", ["bash", "-lc", "g++ --version | head -1"]),
        run_inventory_command("cmake", ["bash", "-lc", "cmake --version | head -1"]),
    ]
    write_text(path, "\n".join(content))


def configure_and_build(run_dir: Path, commands_file: Path, skip_build: bool) -> None:
    if skip_build and BINARY.is_file():
        print(f"[BUILD] skipped; using {BINARY}", flush=True)
        return

    configure_log = run_dir / "logs" / "build" / "configure.log"
    build_log = run_dir / "logs" / "build" / "build.log"
    configure_command = [
        "cmake",
        "-S",
        str(SPATIAL_ROOT / "kde_cuda"),
        "-B",
        str(BUILD_DIR),
        f"-DCMAKE_BUILD_TYPE={BUILD_TYPE}",
        f"-DCUDA_DEVICE_VERSION={CUDA_ARCH}",
    ]
    build_command = [
        "cmake",
        "--build",
        str(BUILD_DIR),
        "--target",
        "kde_cuda_kdtr",
        "-j",
        str(BUILD_JOBS),
    ]

    print("[BUILD] configuring SpatialKDV exact-KDV target", flush=True)
    rc, _, log_text, timed_out = run_command(
        configure_command,
        configure_log,
        commands_file,
        CONFIGURE_TIMEOUT_SECONDS,
    )
    if timed_out or rc != "0":
        raise RuntimeError("SpatialKDV CMake configure failed:\n" + log_text)

    print("[BUILD] building exact-KDV adapted kde_cuda_kdtr", flush=True)
    rc, _, log_text, timed_out = run_command(
        build_command,
        build_log,
        commands_file,
        BUILD_TIMEOUT_SECONDS,
    )
    if timed_out or rc != "0":
        raise RuntimeError("SpatialKDV build failed:\n" + log_text)
    if not BINARY.is_file():
        raise FileNotFoundError(BINARY)


def workload_query_count(workload: Workload) -> int:
    return workload.grid_rows * workload.grid_cols


def timeout_for_workload(workload: Workload) -> int:
    return SMOKE_TIMEOUT_SECONDS if workload.run_group == "smoke" else KDV_TIMEOUT_SECONDS


def workload_command(workload: Workload, output_path: Path) -> list[str]:
    return [
        str(BINARY),
        str(workload.data_path),
        str(workload.grid_rows),
        str(workload.grid_cols),
        "--scott-diag",
        workload.scale_value,
        str(output_path),
    ]


def output_name(workload: Workload) -> str:
    safe_b = workload.scale_value.replace(".", "p")
    return (
        f"{workload.workload}_spatialkdv_fp32_"
        f"{workload.grid_rows}x{workload.grid_cols}_scott_diag_b{safe_b}.out"
    )


def run_workload(workload: Workload, run_dir: Path, commands_file: Path) -> RunRecord:
    output_path = run_dir / "outputs" / "fp32" / workload.run_group / output_name(workload)
    log_path = run_dir / "logs" / workload.run_group / f"{output_path.stem}.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = workload_command(workload, output_path)
    timeout_seconds = timeout_for_workload(workload)

    print(f"[RUN] FP32 {workload.run_group} {workload.workload} (timeout={timeout_seconds}s)", flush=True)
    return_code, wall_seconds, log_text, timed_out = run_command(
        command,
        log_path,
        commands_file,
        timeout_seconds,
    )
    metrics = parse_metrics(log_text)
    status = "timeout" if timed_out else ("ok" if return_code == "0" else f"failed({return_code})")
    if status == "ok" and metrics.get("timing_scope") != EXPECTED_TIMING_SCOPE:
        status = "missing_or_bad_timing"
    if status == "ok" and metrics.get("query_count") != str(workload_query_count(workload)):
        status = "bad_query_count"
    if status == "ok" and metrics.get("precision") != PRECISION:
        status = "bad_precision"

    return RunRecord(
        workload=workload,
        output_path=output_path,
        log_path=log_path,
        command=command,
        return_code=return_code,
        timeout_seconds=timeout_seconds,
        cli_wall_seconds=wall_seconds,
        timing_scope=metrics.get("timing_scope", ""),
        execution_seconds=metrics.get("execution_seconds", ""),
        query_count_from_binary=metrics.get("query_count", ""),
        qps=metrics.get("qps", ""),
        status=status,
    )


def summary_row(
    record: RunRecord,
    timestamp: str,
    git_branch: str,
    git_commit: str,
    gpu_info: dict[str, str],
) -> dict[str, str]:
    data_rows, dimension = matrix_shape(record.workload.data_path)
    return {
        "timestamp": timestamp,
        "machine": MACHINE,
        "hostname": socket.gethostname(),
        "run_group": record.workload.run_group,
        "method": METHOD,
        "method_token": METHOD_TOKEN,
        "engine": ENGINE,
        "backend": BACKEND,
        "precision": PRECISION,
        "workload": record.workload.workload,
        "dataset": record.workload.dataset,
        "mode": "kdv",
        "data_path": str(record.workload.data_path),
        "query_path": "",
        "grid_rows": str(record.workload.grid_rows),
        "grid_cols": str(record.workload.grid_cols),
        "data_rows": str(data_rows),
        "query_rows": str(workload_query_count(record.workload)),
        "binary_query_count": record.query_count_from_binary,
        "dimension": str(dimension),
        "weights": "none",
        "scale_mode": record.workload.scale_mode,
        "scale_value": record.workload.scale_value,
        "scale": f"{record.workload.scale_mode} b={record.workload.scale_value}",
        "timing_scope": record.timing_scope,
        "runtime_s": record.execution_seconds,
        "qps": record.qps,
        "timeout_seconds": str(record.timeout_seconds),
        "cli_wall_seconds": f"{record.cli_wall_seconds:.9f}",
        "output_path": str(record.output_path),
        "log_path": str(record.log_path),
        "command": shell_join(record.command),
        "status": record.status,
        "return_code": str(record.return_code),
        "build_type": BUILD_TYPE,
        "cuda_arch": CUDA_ARCH,
        "gpu_name": gpu_info["gpu_name"],
        "gpu_memory_total": gpu_info["gpu_memory_total"],
        "gpu_driver_version": gpu_info["gpu_driver_version"],
        "gpu_compute_cap": gpu_info["gpu_compute_cap"],
        "git_branch": git_branch,
        "git_commit": git_commit,
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def correctness_row(
    timestamp: str,
    git_branch: str,
    git_commit: str,
    workload: Workload,
    reference_method: str,
    reference_output: Path,
    record: RunRecord,
) -> dict[str, str]:
    if not record.output_path.is_file() or not reference_output.is_file():
        metrics = {
            "value_count": "",
            "max_abs_err": "",
            "max_rel_err": "",
            "failure_count": "",
            "tolerance": f"abs={FP32_ABS_TOLERANCE:g};rel={FP32_REL_TOLERANCE:g}",
            "status": "missing_output",
        }
    else:
        metrics = compare_numeric_files(record.output_path, reference_output)

    return {
        "timestamp": timestamp,
        "machine": MACHINE,
        "run_group": workload.run_group,
        "workload": workload.workload,
        "mode": "kdv",
        "scale": f"{workload.scale_mode} b={workload.scale_value}",
        "reference_method": reference_method,
        "method": METHOD,
        "precision": PRECISION,
        "reference_output": str(reference_output),
        "output_path": str(record.output_path),
        "value_count": metrics["value_count"],
        "max_abs_err": metrics["max_abs_err"],
        "max_rel_err": metrics["max_rel_err"],
        "failure_count": metrics["failure_count"],
        "tolerance": metrics["tolerance"],
        "status": metrics["status"],
        "git_branch": git_branch,
        "git_commit": git_commit,
    }


def full_reference_for(workload: Workload) -> Path:
    safe_b = workload.scale_value.replace(".", "p")
    return (
        BASIC_GROUND_TRUTH_ROOT
        / f"{workload.workload}_basic-scan_fp64_{workload.grid_rows}x{workload.grid_cols}_scott_diag_b{safe_b}.out"
    )


def main() -> int:
    script_name = Path(sys.argv[0]).name
    smoke_only = False
    skip_build = False
    for arg in sys.argv[1:]:
        if arg in ("-h", "--help"):
            print(
                f"usage: {script_name} [--smoke-only] [--skip-build]\n\n"
                "Builds the SpatialKDV exact-KDV target, runs the smoke\n"
                "test, and unless --smoke-only is set, runs supported full KDV\n"
                "workloads. Outputs are written under tmp-results/stage0/spatialkdv/."
            )
            return 0
        if arg == "--smoke-only":
            smoke_only = True
            continue
        if arg == "--skip-build":
            skip_build = True
            continue
        print(f"Unknown argument: {arg}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = TMP_RESULT_ROOT / f"{MACHINE}_{timestamp}"
    commands_file = run_dir / "commands.sh"
    inventory_file = run_dir / "machine_inventory.txt"
    summary_file = run_dir / "summary.csv"
    correctness_file = run_dir / "correctness.csv"

    run_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        commands_file,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f"# Generated by {script_name} at {timestamp}.\n\n",
    )
    commands_file.chmod(0o755)

    required_files = [
        SPATIAL_ROOT / "kde_cuda/kde_cuda_kdtr.cu",
        SPATIAL_ROOT / "kde_cuda/kde_kernel_kdtr.cu",
        SMOKE_SUSY_VIS_X,
        SMOKE_KDV_REF,
    ]
    if not smoke_only:
        required_files.extend(
            [
                HOME_VIS_X,
                SUSY_VIS_X,
                BASIC_GROUND_TRUTH_ROOT / "kdv_home_visual_basic-scan_fp64_1920x2560_scott_diag_b1.out",
                BASIC_GROUND_TRUTH_ROOT / "kdv_susy_visual_basic-scan_fp64_1920x2560_scott_diag_b1.out",
            ]
        )
    require_files(required_files)

    git_branch = run_shell_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_commit = run_shell_capture(["git", "rev-parse", "HEAD"])
    gpu_info = collect_gpu_summary()
    write_inventory(inventory_file, timestamp, git_branch, git_commit)
    configure_and_build(run_dir, commands_file, skip_build)

    smoke_workloads = [
        Workload(
            "smoke",
            "smoke_kdv_susy_vis_10000_256x256",
            "SUSY visualization validation subset",
            SMOKE_SUSY_VIS_X,
            SMOKE_KDV_ROWS,
            SMOKE_KDV_COLS,
            "scott_diag",
            SMOKE_SCOTT_B,
        ),
    ]
    full_workloads = [
        Workload("full", "kdv_home_visual", "Home visualization", HOME_VIS_X, KDV_ROWS, KDV_COLS, "scott_diag", KDV_SCOTT_B),
        Workload("full", "kdv_susy_visual", "SUSY visualization", SUSY_VIS_X, KDV_ROWS, KDV_COLS, "scott_diag", KDV_SCOTT_B),
    ]

    workloads = smoke_workloads if smoke_only else [*smoke_workloads, *full_workloads]
    records = [run_workload(workload, run_dir, commands_file) for workload in workloads]

    summary_fields = [
        "timestamp",
        "machine",
        "hostname",
        "run_group",
        "method",
        "method_token",
        "engine",
        "backend",
        "precision",
        "workload",
        "dataset",
        "mode",
        "data_path",
        "query_path",
        "grid_rows",
        "grid_cols",
        "data_rows",
        "query_rows",
        "binary_query_count",
        "dimension",
        "weights",
        "scale_mode",
        "scale_value",
        "scale",
        "timing_scope",
        "runtime_s",
        "qps",
        "timeout_seconds",
        "cli_wall_seconds",
        "output_path",
        "log_path",
        "command",
        "status",
        "return_code",
        "build_type",
        "cuda_arch",
        "gpu_name",
        "gpu_memory_total",
        "gpu_driver_version",
        "gpu_compute_cap",
        "git_branch",
        "git_commit",
    ]
    write_csv(
        summary_file,
        [summary_row(record, timestamp, git_branch, git_commit, gpu_info) for record in records],
        summary_fields,
    )

    record_by_key = {(record.workload.run_group, record.workload.workload): record for record in records}
    correctness_rows = []
    for workload in smoke_workloads:
        if (workload.run_group, workload.workload) in record_by_key:
            correctness_rows.append(
                correctness_row(
                    timestamp,
                    git_branch,
                    git_commit,
                    workload,
                    "checked_results",
                    SMOKE_KDV_REF,
                    record_by_key[(workload.run_group, workload.workload)],
                )
            )
    if not smoke_only:
        for workload in full_workloads:
            correctness_rows.append(
                correctness_row(
                    timestamp,
                    git_branch,
                    git_commit,
                    workload,
                    "Basic-Scan",
                    full_reference_for(workload),
                    record_by_key[(workload.run_group, workload.workload)],
                )
            )

    correctness_fields = [
        "timestamp",
        "machine",
        "run_group",
        "workload",
        "mode",
        "scale",
        "reference_method",
        "method",
        "precision",
        "reference_output",
        "output_path",
        "value_count",
        "max_abs_err",
        "max_rel_err",
        "failure_count",
        "tolerance",
        "status",
        "git_branch",
        "git_commit",
    ]
    write_csv(correctness_file, correctness_rows, correctness_fields)

    run_failures = [record for record in records if record.status != "ok"]
    correctness_failures = [row for row in correctness_rows if row["status"] != "ok"]

    print(f"[DONE] run_dir={run_dir}")
    print(f"[DONE] summary={summary_file}")
    print(f"[DONE] correctness={correctness_file}")
    print("[DONE] Persistent promotion is manual after review.")

    if run_failures or correctness_failures:
        print(
            f"[WARN] run_failures={len(run_failures)} "
            f"correctness_failures={len(correctness_failures)}",
            file=sys.stderr,
        )
    if run_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
