# MSF footprints demo — HerdNetSeg inference in one container.
# Base = the same PyTorch/CUDA image the model was trained in on carrot (logs/herdnet_camp_v3.sh),
# so the inference numerics match the validation report. CUDA 12.4 runs on an A10G (g5.4xlarge)
# and on consumer cards; without a GPU the same image falls back to CPU (--device auto).
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# git: pip install from GitHub. libgl1/libglib2.0/libxcb1: opencv (a dependency of animaloc) links them
# even in a headless container.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libxcb1 \
 && rm -rf /var/lib/apt/lists/*

# HerdNetPlus is pinned to the commit the shipped model was trained with; bump HERDNET_REF deliberately.
ARG HERDNET_REF=1e35d666a942607daa26fe3bc71ed75b008ffcd9
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir "git+https://github.com/cwinkelmann/HerdNetPlus.git@${HERDNET_REF}" \
 && python -c "import torch, animaloc; from animaloc.models import HerdNetSeg; import rasterio, geopandas, skimage, matplotlib; print('ok torch', torch.__version__, 'cuda build', torch.version.cuda)"

# Model weights. models/<name>/{best_model.pth, stretch.json, .hydra/config.yaml}.
# Put the run directory under ./models before `docker build` to bake it into the image (self-contained
# delivery), or leave the folder empty and mount it at run time (run.sh: MODEL_DIR=...).
COPY models/ /models/
# MPLCONFIGDIR/HF_HOME/TORCH_HOME under /tmp: writable when the container runs as an arbitrary uid (run.sh -u).
ENV MODEL_DIR=/models/seghead_joint_all HERDNET_REF=${HERDNET_REF} MPLCONFIGDIR=/tmp/mpl HF_HOME=/tmp/hf TORCH_HOME=/tmp/torch

COPY contexts.json /app/contexts.json
COPY app/ /app/
WORKDIR /app
EXPOSE 7860
# GUI instead of the CLI:  docker run --rm --gpus all -p 7860:7860 --entrypoint python msf-footprints:0.1 gui.py
ENTRYPOINT ["python", "-u", "/app/run.py"]
CMD ["--help"]
