#!/usr/bin/env bash
# Idempotent dependency setup for the EGVSR-PyTorch Cloud Agent environment.
# Installs a CPU build of PyTorch plus the project's Python dependencies into a
# local virtualenv (.venv) using uv. Safe to re-run.
set -euo pipefail

# ---- locate repo root (parent of this .cursor/ dir) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- ensure uv is available ----
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

# ---- create the virtualenv (idempotent) ----
if [ ! -x ".venv/bin/python" ]; then
  uv venv --python 3.12 .venv
fi

# ---- install dependencies ----
# CPU PyTorch (this environment has no GPU) resolved together with the rest so
# the dependency set stays consistent.
uv pip install --python .venv --torch-backend cpu \
  torch==2.13.0 torchvision==0.28.0 \
  "numpy<2.6" \
  opencv-python-headless \
  pyyaml \
  scipy \
  scikit-image \
  tqdm \
  lmdb \
  matplotlib \
  flow_vis

echo ""
echo "EGVSR environment ready. Activate it with:  source .venv/bin/activate"
