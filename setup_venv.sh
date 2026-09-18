#!/usr/bin/env bash
# No-Docker setup: a Python venv with CUDA torch + the geo stack + HerdNetPlus (pinned) + gradio.
#   ./setup_venv.sh            # creates ./.venv ; then:  .venv/bin/python app/gui.py
# CPU-only machine: CUDA_INDEX=https://download.pytorch.org/whl/cpu ./setup_venv.sh
# Uses uv when present (it also fetches Python 3.11 if the system lacks it), else python3 -m venv.
set -euo pipefail
cd "$(dirname "$0")"
HERDNET_REF="${HERDNET_REF:-1e35d666a942607daa26fe3bc71ed75b008ffcd9}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null; then
  [ -d .venv ] || uv venv --python 3.11 .venv
  PIP=(uv pip install --python .venv/bin/python)
else
  [ -d .venv ] || python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  PIP=(.venv/bin/python -m pip install)
fi
"${PIP[@]}" "torch==2.4.*" torchvision --index-url "$CUDA_INDEX"
"${PIP[@]}" -r requirements.txt gradio
"${PIP[@]}" "git+https://github.com/cwinkelmann/HerdNetPlus.git@${HERDNET_REF}"
.venv/bin/python -c "import torch, gradio; from animaloc.models import HerdNetSeg; import rasterio, geopandas; print('ok torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gradio', gradio.__version__)"
echo "venv ready: .venv/bin/python app/gui.py"
