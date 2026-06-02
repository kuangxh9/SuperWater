#!/usr/bin/env bash
#
# One-shot installer for the SuperWater environment.
#
# Creates the `superwater` conda environment (PyTorch 2.5.1 + CUDA 11.8, e3nn 0.5.4,
# rdkit, openbabel, ...), installs the CUDA-specific PyTorch Geometric extension
# wheels from the PyG wheel index, and installs the `superwater` package itself
# (editable) together with its console-script entry points.
#
# Usage:
#     bash scripts/install.sh
#
# Requirements: a working conda/mamba installation and an NVIDIA GPU with a driver
# that supports CUDA 11.8. Run from the repository root.

set -euo pipefail

ENV_NAME="superwater"
# PyG wheel index matching the conda PyTorch/CUDA build (PyTorch 2.5.x + CUDA 11.8).
PYG_INDEX="https://data.pyg.org/whl/torch-2.5.0+cu118.html"

# Resolve the directory of this script so the installer works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Prefer mamba if available (much faster solver), otherwise fall back to conda.
if command -v mamba >/dev/null 2>&1; then
    CONDA="mamba"
else
    CONDA="conda"
fi

echo ">> [1/4] Creating conda environment '${ENV_NAME}' from environment.yml ..."
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "   Environment '${ENV_NAME}' already exists; updating it instead."
    ${CONDA} env update -n "${ENV_NAME}" -f environment.yml --prune
else
    ${CONDA} env create -f environment.yml
fi

# Make `conda activate` available inside this non-interactive shell.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo ">> [2/4] Installing PyTorch Geometric CUDA wheels from ${PYG_INDEX} ..."
pip install -r requirements-pyg-cu118.txt

echo ">> [3/4] Installing the superwater package (editable) ..."
pip install -e .

echo ">> [4/4] Verifying the installation ..."
python - <<'PY'
import torch
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
import e3nn, torch_cluster, torch_scatter, torch_geometric  # noqa: F401
import superwater  # noqa: F401
print("superwater and all core dependencies import successfully.")
PY

cat <<EOF

Done. Activate the environment with:

    conda activate ${ENV_NAME}

Optional: clone Meta's ESM repo into ./esm to generate ESM-2 embeddings:

    git clone https://github.com/facebookresearch/esm.git
EOF
