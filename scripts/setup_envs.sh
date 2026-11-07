#!/usr/bin/env bash
# Creates the two conda envs on the AWS GPU instance.
#   bash scripts/setup_envs.sh train     # mamba-train  (Python 3.9, torch 2.0.1 cu118)
#   bash scripts/setup_envs.sh teacher   # whisper-teacher (Python 3.10, recent torch)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"

case "${1:-}" in
train)
  conda create -y -n mamba-train python=3.9
  conda activate mamba-train
  # nvcc 11.8 is only needed if the mamba/causal-conv1d wheels must be compiled
  conda install -y -c nvidia cuda-nvcc=11.8 cuda-cudart-dev=11.8 || true
  pip install --upgrade pip wheel packaging ninja
  # 1) PyTorch first
  pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  pip install "numpy<2"          # torch 2.0.1 is not compatible with numpy 2
  # 2) causal-conv1d, 3) mamba-ssm  (pip fetches a prebuilt wheel when one matches,
  #    otherwise compiles: 10-30 min)
  pip install causal-conv1d==1.1.3.post1 --no-build-isolation
  pip install mamba-ssm==1.1.3.post1 --no-build-isolation
  pip install speechbrain==1.0.0 sentencepiece soundfile librosa tqdm wandb \
              num2words jiwer pytest matplotlib "numpy<2"
  python "$REPO/scripts/check_install.py"
  ;;
teacher)
  conda create -y -n whisper-teacher python=3.10
  conda activate whisper-teacher
  pip install --upgrade pip
  pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -r "$REPO/teacher/requirements.txt"
  # CTranslate2 (faster-whisper) loads cuBLAS/cuDNN from these pip wheels:
  SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
  mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
  echo "export LD_LIBRARY_PATH=$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-}" \
      > "$CONDA_PREFIX/etc/conda/activate.d/cuda_libs.sh"
  ;;
*)
  echo "usage: $0 train|teacher"; exit 1;;
esac
