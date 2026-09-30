#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Install the CUDA / OpenMMLab stack BEVision needs, in the one order that works.
#
# Usage:  bash scripts/install_mmdet3d.sh
#
# Tested on: Python 3.12, CUDA 11.8, Tesla T4.
#
# Why a script and not just requirements-cuda.txt: mmcv wheels are compiled against
# BOTH a torch version and a CUDA version, mmdet must be installed with --no-deps,
# and mmdetection3d is not on PyPI at all. Each of those steps fails silently or
# half-succeeds if the order is wrong, so the order is enforced here.
# ---------------------------------------------------------------------------
set -euo pipefail

PY_VERSION="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "==> Python ${PY_VERSION}"
if [[ "${PY_VERSION}" > "3.12" ]]; then
    echo "ERROR: mmcv 2.2.0 has no cp313 wheel and Python 3.13 removed distutils." >&2
    echo "       Use a Python 3.11 or 3.12 runtime." >&2
    exit 1
fi

# --- 0. Version gates in mmdet/mmdet3d are check-then-assert at import time -----
# They compare the INSTALLED mmcv against a declared maximum using a strict '<'.
# With mmcv == the declared max, the assertion fails. We relax it after install,
# in step 4, against the real installed files.

# --- 1. setuptools first: older versions break the mmcv wheel build -----------
echo "==> setuptools"
python -m pip install --quiet --upgrade setuptools

# --- 2. torch, pinned: mmcv's wheel below is built against exactly this ---------
echo "==> torch 2.2.0 + torchvision 0.17.0 (cu118)"
python -m pip install --quiet \
    torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu118

# --- 3. OpenMMLab -------------------------------------------------------------
echo "==> mmengine / mmcv 2.2.0"
python -m pip install --quiet mmengine
python -m pip install --quiet mmcv==2.2.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.2.0/index.html

# --no-deps: mmdet's declared numpy constraint pulls numpy 2.x, which mmdetection3d
# cannot use. Its real runtime deps are installed explicitly below.
echo "==> mmdet (--no-deps) and its skipped runtime deps"
python -m pip install --quiet "mmdet>=3.2.0" --no-deps
python -m pip install --quiet terminaltables "transformers<4.50" --no-deps

# --- 4. mmdet3d is not on PyPI: clone and install ------------------------------
echo "==> mmdetection3d (from source)"
MMDET3D_DIR="${MMDET3D_DIR:-mmdetection3d}"
if [[ ! -d "${MMDET3D_DIR}" ]]; then
    git clone --depth 1 https://github.com/open-mmlab/mmdetection3d.git "${MMDET3D_DIR}"
fi
python -m pip install --quiet "./${MMDET3D_DIR}" --no-build-isolation --no-deps

# --- 5. Relax the mmcv upper-bound assert in mmdet and mmdet3d -----------------
# Both do: assert mmcv_version < digit_version(mmcv_maximum_version)
# With mmcv 2.2.0 installed and 2.2.0 declared, that is 2.2.0 < 2.2.0 -> False.
# Located via find_spec (NOT import -- importing is what crashes) so it works on
# any Python version and any site-packages layout.
echo "==> relaxing mmcv version gates"
python - <<'PY'
import importlib.util
import re

for package in ("mmdet", "mmdet3d"):
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        print(f"    {package}: not installed, skipping")
        continue
    with open(spec.origin, encoding="utf-8") as handle:
        source = handle.read()
    patched, count = re.subn(
        r"mmcv_maximum_version\s*=\s*'[\d.]+'",
        "mmcv_maximum_version = '2.3.0'",
        source,
    )
    if count:
        with open(spec.origin, "w", encoding="utf-8") as handle:
            handle.write(patched)
        print(f"    {package}: relaxed {count} gate(s) in {spec.origin}")
    else:
        print(f"    {package}: no version gate found (already relaxed or changed layout)")
PY

# --- 6. numpy last: torch/mmcv/mmdet all try to upgrade it ---------------------
echo "==> numpy < 2.0 (force-reinstall, must come last)"
python -m pip install --quiet --no-cache-dir --force-reinstall "numpy<2.0.0"

# --- 7. verify -----------------------------------------------------------------
echo "==> verifying imports"
python - <<'PY'
import numpy
import torch

print(f"    numpy : {numpy.__version__}")
print(f"    torch : {torch.__version__}")
print(f"    cuda  : {torch.cuda.is_available()}")
import mmcv
import mmdet
import mmdet3d

print(f"    mmcv  : {mmcv.__version__}")
print(f"    mmdet : {mmdet.__version__}")
print(f"    mmdet3d: {mmdet3d.__version__}")
from mmdet3d.apis import inference_detector, init_model  # noqa: F401

print("    OK")
PY

echo
echo "Done. NOTE: step 5 edits site-packages. Re-run this script after any runtime"
echo "restart, or the import of mmdet/mmdet3d will fail again."
