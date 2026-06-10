#!/usr/bin/env python3
"""Run the Zhang et al. 2017 CUDA KDE code on a KARL-style 2D KDV input.

This is a throughput adapter for the original raster/grid KDE implementation.
The adapted CUDA kernel computes the KARL-like unnormalized Gaussian sum on raw
coordinates using bandwidth coefficients inside the kernel.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    data_root = Path(
        os.environ.get(
            "GPU_KDE_DATA_ROOT",
            "/home/ubuntu/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Estimation",
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="2D KARL .data file")
    parser.add_argument("--dataset", default=None, help="Dataset label for summary output")
    parser.add_argument("--rows", type=int, default=1920, help="Raster/query-grid rows")
    parser.add_argument("--cols", type=int, default=2560, help="Raster/query-grid columns")
    parser.add_argument(
        "--coordinate-mode",
        choices=("pixel", "raw"),
        default="raw",
        help="raw keeps original coordinates; pixel rescales data to [0, cols-1] x [0, rows-1] before the kernel computes bandwidth",
    )
    parser.add_argument(
        "--h-option",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Original Zhang17 preprocessing mode; keep 0 for KARL-like KDV timing runs",
    )
    parser.add_argument("--max-points", type=int, default=None, help="Optional prefix sample for smoke tests")
    parser.add_argument("--repo-dir", type=Path, default=repo_dir, help="GPUParallelSpatialAdaptiveKDE repo dir")
    parser.add_argument("--build-dir", type=Path, default=repo_dir / "build", help="CMake build dir")
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory for converted inputs/logs")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional CSV to append one summary row")
    parser.add_argument("--skip-build", action="store_true", help="Do not run CMake build")
    parser.add_argument("--write-output", action="store_true", help="Keep the output ASC raster")
    parser.add_argument("--cuda-device-version", default="86", help="CMake CUDA_DEVICE_VERSION")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=data_root / "experiment_results",
        help="Default parent for run dirs when --run-dir is omitted",
    )
    return parser.parse_args()


def read_shape_and_bounds(path: Path, max_points: int | None) -> tuple[int, int, list[float], list[float]]:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().split()
        if len(header) != 2:
            raise ValueError(f"Invalid .data header in {path}")
        n_rows, dim = int(header[0]), int(header[1])
        if dim != 2:
            raise ValueError(f"Zhang17 adapter only supports 2D inputs, got dim={dim}")

        limit = min(n_rows, max_points) if max_points else n_rows
        mins = [math.inf, math.inf]
        maxs = [-math.inf, -math.inf]
        count = 0
        for line in f:
            if count >= limit:
                break
            if not line.strip():
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 2:
                raise ValueError(f"Expected 2 values on data row {count + 2}")
            for i, val in enumerate(vals):
                mins[i] = min(mins[i], val)
                maxs[i] = max(maxs[i], val)
            count += 1
    if count == 0:
        raise ValueError(f"No points found in {path}")
    return count, dim, mins, maxs


def write_points_csv(
    src: Path,
    dst: Path,
    count: int,
    mins: list[float],
    maxs: list[float],
    rows: int,
    cols: int,
    coordinate_mode: str,
) -> None:
    row_span = maxs[0] - mins[0]
    col_span = maxs[1] - mins[1]
    if row_span == 0 or col_span == 0:
        raise ValueError("Cannot rescale degenerate 2D data bounds")

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8", newline="") as fout:
        fin.readline()
        writer = csv.writer(fout)
        writer.writerow(["x", "y"])
        written = 0
        for line in fin:
            if written >= count:
                break
            if not line.strip():
                continue
            row_v, col_v = (float(x) for x in line.split())
            if coordinate_mode == "pixel":
                x = (col_v - mins[1]) / col_span * float(cols - 1)
                y = (row_v - mins[0]) / row_span * float(rows - 1)
            else:
                x = col_v
                y = row_v
            writer.writerow([f"{x:.9g}", f"{y:.9g}"])
            written += 1


def write_mask(path: Path, rows: int, cols: int, mins: list[float], maxs: list[float], coordinate_mode: str) -> float:
    if coordinate_mode == "pixel":
        xll = -0.5
        yll = -0.5
        cell_size = 1.0
    else:
        row_cell = (maxs[0] - mins[0]) / max(rows - 1, 1)
        col_cell = (maxs[1] - mins[1]) / max(cols - 1, 1)
        cell_size = max(row_cell, col_cell)
        xll = mins[1] - 0.5 * cell_size
        yll = mins[0] - 0.5 * cell_size

    row_text = " ".join(["0"] * cols)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"ncols         {cols}\n")
        f.write(f"nrows         {rows}\n")
        f.write(f"xllcorner     {xll:.12g}\n")
        f.write(f"yllcorner     {yll:.12g}\n")
        f.write(f"cellsize      {cell_size:.12g}\n")
        f.write("NODATA_value  -9999\n")
        for _ in range(rows):
            f.write(row_text)
            f.write("\n")
    return cell_size


def run_checked(cmd: list[str], cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def build(repo_dir: Path, build_dir: Path, cuda_device_version: str) -> Path:
    run_checked(["cmake", "-S", str(repo_dir / "kde_cuda"), "-B", str(build_dir), f"-DCUDA_DEVICE_VERSION={cuda_device_version}"])
    run_checked(["cmake", "--build", str(build_dir), "-j"])
    binary = build_dir / "kde_cuda_kdtr"
    if not binary.exists():
        raise FileNotFoundError(binary)
    return binary


def parse_timing(stdout: str) -> tuple[float | None, float | None]:
    exclusive = None
    inclusive = None
    for line in stdout.splitlines():
        m = re.search(r"Computation on GPU took\s+([0-9.]+)\s+ms\s+\((EXCLUSIVE|INCLUSIVE)\)", line)
        if m:
            val_s = float(m.group(1)) / 1000.0
            if m.group(2) == "EXCLUSIVE":
                exclusive = val_s
            else:
                inclusive = val_s
    return exclusive, inclusive


def append_summary(path: Path, row: dict[str, object]) -> None:
    fields = [
        "case_id",
        "mode",
        "dataset",
        "coordinate_mode",
        "h_option",
        "resolution",
        "point_count",
        "query_count",
        "gpu_exclusive_s",
        "gpu_inclusive_s",
        "qps_exclusive",
        "qps_inclusive",
        "status",
        "log_file",
        "output_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()

    dataset = args.dataset or args.data.stem
    max_tag = f"_n{args.max_points}" if args.max_points else ""
    case_id = f"zhang17_{dataset}_{args.coordinate_mode}_r{args.rows}x{args.cols}_hopt{args.h_option}{max_tag}"

    run_dir = args.run_dir
    if run_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = args.result_root / f"zhang17_kdv_{ts}" / case_id
    run_dir.mkdir(parents=True, exist_ok=True)

    binary = args.build_dir / "kde_cuda_kdtr"
    if not args.skip_build or not binary.exists():
        binary = build(args.repo_dir, args.build_dir, args.cuda_device_version)

    point_count, _, mins, maxs = read_shape_and_bounds(args.data, args.max_points)
    points_csv = run_dir / "points.csv"
    mask_asc = run_dir / "mask.asc"
    log_file = run_dir / "run.log"
    output_file = run_dir / "density.asc"

    print(f"[info] Converting {point_count} points from {args.data}", flush=True)
    write_points_csv(args.data, points_csv, point_count, mins, maxs, args.rows, args.cols, args.coordinate_mode)
    cell_size = write_mask(mask_asc, args.rows, args.cols, mins, maxs, args.coordinate_mode)
    print(f"[info] Wrote converted inputs to {run_dir} (cell_size={cell_size:g})", flush=True)

    cmd = [
        str(binary),
        "1",
        str(points_csv),
        str(mask_asc),
        str(args.h_option),
        "1",
        "0",
        "-",
        str(output_file),
    ]

    print("[cmd]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=run_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_file.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, end="")
    if not args.write_output and output_file.exists():
        output_file.unlink()

    exclusive_s, inclusive_s = parse_timing(proc.stdout)
    query_count = args.rows * args.cols
    qps_ex = query_count / exclusive_s if exclusive_s and exclusive_s > 0 else None
    qps_in = query_count / inclusive_s if inclusive_s and inclusive_s > 0 else None
    status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"

    print(
        f"[summary] case_id={case_id} status={status} "
        f"exclusive_s={exclusive_s} inclusive_s={inclusive_s} "
        f"qps_exclusive={qps_ex} qps_inclusive={qps_in}",
        flush=True,
    )

    row = {
        "case_id": case_id,
        "mode": "kdv",
        "dataset": dataset,
        "coordinate_mode": args.coordinate_mode,
        "h_option": args.h_option,
        "resolution": f"{args.rows}x{args.cols}",
        "point_count": point_count,
        "query_count": query_count,
        "gpu_exclusive_s": exclusive_s if exclusive_s is not None else "",
        "gpu_inclusive_s": inclusive_s if inclusive_s is not None else "",
        "qps_exclusive": qps_ex if qps_ex is not None else "",
        "qps_inclusive": qps_in if qps_in is not None else "",
        "status": status,
        "log_file": str(log_file),
        "output_file": str(output_file if args.write_output else ""),
    }
    if args.summary_csv:
        append_summary(args.summary_csv, row)

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
