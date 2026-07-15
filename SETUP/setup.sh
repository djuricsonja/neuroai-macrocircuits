#!/usr/bin/env bash
# Sets up the project environment with uv (macOS / Linux).
#
# Installs uv if it is missing, creates .venv with the Python version pinned in
# .python-version, and installs the locked dependencies from requirements.lock.
# Safe to re-run: it converges the environment to the lock file.
#
#   ./SETUP/setup.sh          # CPU torch
#   ./SETUP/setup.sh --cuda   # CUDA torch
set -euo pipefail

cuda=0
[ "${1:-}" = "--cuda" ] && cuda=1

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found - installing it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer updates PATH for new shells only, so extend it for this one.
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv $(uv --version)"

cd "$repo"

# uv venv reads .python-version and downloads that interpreter if it is missing.
# It refuses to write into an existing directory unless told how to treat it, so
# reuse a healthy .venv and rebuild one left half-written by an interrupted run.
if [ -d .venv ] && [ ! -x .venv/bin/python ]; then
    echo "Existing .venv is incomplete - recreating it..."
    uv venv --clear
else
    uv venv --allow-existing
fi

uv pip sync SETUP/requirements.lock

if [ "$cuda" = "1" ]; then
    echo "Installing a CUDA build of PyTorch..."
    uv pip install torch --torch-backend=auto
fi

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "In VS Code, select .venv as the notebook kernel."
