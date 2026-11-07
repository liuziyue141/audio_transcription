"""Shared fixtures. Environment variables used by the data/GPU tests:

  WORK_DIR        e.g. /data/work  (prepared CSVs, tokenizer, teacher features)
  TEACHER_SUFFIX  "" for the full extraction, "_limit500" for the timing run
  RUN_DIR         a training output folder (for checks 5-trained and 11)
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
YAML = os.path.join(REPO, "hparams", "CTC", "conmamba_causal_distill.yaml")


def env_dir(name):
    v = os.environ.get(name)
    if not v or not os.path.isdir(v):
        pytest.skip(f"set {name} to run this test")
    return v


@pytest.fixture
def work_dir():
    return env_dir("WORK_DIR")


@pytest.fixture
def run_dir():
    return env_dir("RUN_DIR")


def require_cuda_mamba():
    torch = pytest.importorskip("torch")
    pytest.importorskip("speechbrain")
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (mamba_ssm kernels)")
    return torch


def build_random_model(tmp_path):
    """Instantiate the causal student from the YAML with random weights."""
    from hyperpyyaml import load_hyperpyyaml

    with open(YAML) as f:
        hp = load_hyperpyyaml(f, {"work_dir": str(tmp_path), "data_folder": "unused"})
    return hp
