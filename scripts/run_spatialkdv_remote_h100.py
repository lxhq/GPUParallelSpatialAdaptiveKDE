#!/usr/bin/env python3
"""Stage 0 remote H100 SpatialKDV runner.

This script reuses the local runner logic and overrides only the machine-
specific paths/knobs. Generated evidence is written to remote tmp-results and
must be copied back to local persistent storage manually after review.
"""

from __future__ import annotations

from pathlib import Path

import run_spatialkdv_local_3080ti as runner


runner.SPATIAL_ROOT = Path("/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/baselines/GPUParallelSpatialAdaptiveKDE")
runner.BUILD_DIR = runner.SPATIAL_ROOT / "build-stage0-remote-h100"
runner.BINARY = runner.BUILD_DIR / "kde_cuda_kdtr"
runner.CUDA_HOME = Path("/usr/local/cuda-12.4")

runner.DATA_ROOT = Path("/home/lxheq/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation")
runner.TMP_RESULT_ROOT = Path("/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/tmp-results/stage0/spatialkdv")
runner.PERSISTENT_RESULT_ROOT = Path("/home/lxheq/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation/exact/experiments/stage0")
runner.EXACT_REPO_ROOT = Path("/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/GPU-kernel-density-exact")
runner.BASIC_GROUND_TRUTH_ROOT = runner.PERSISTENT_RESULT_ROOT / "ground_truth"

runner.MACHINE = "remote-h100"
runner.CUDA_ARCH = "90"

runner.HOME_VIS_X = runner.DATA_ROOT / "home_visualization/HT_Sensor_dataset_vis_X.data"
runner.SUSY_VIS_X = runner.DATA_ROOT / "susy_visualization/SUSY_vis_X.data"

runner.SMOKE_SUSY_VIS_X = runner.EXACT_REPO_ROOT / "data/SUSY_vis_10000_X.data"
runner.SMOKE_KDV_REF = runner.EXACT_REPO_ROOT / "results/SUSY_vis_10000_kdv_256x256_basic_scan_fp64_scott_diag_b1.out"


if __name__ == "__main__":
    raise SystemExit(runner.main())
