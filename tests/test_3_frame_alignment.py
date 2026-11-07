"""Sanity check 3: student vs teacher frame counts differ by <= 2."""

import csv
import json
import os
import random

import pytest

from distill_utils import (cnn_out_frames, student_frames, student_mel_frames,
                           teacher_frames)


def test_formula_all_lengths():
    """Pure frame math, every length from 1 s to 30 s in 10 ms steps."""
    worst = 0
    for n in range(16000, 30 * 16000 + 1, 160):
        for off in (0, 37, 159):
            d = abs(student_frames(n + off) - teacher_frames(n + off))
            worst = max(worst, d)
    assert worst <= 2, worst


def test_real_cnn_matches_formula(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("speechbrain")
    from conftest import build_random_model

    try:
        hp = build_random_model(tmp_path)
    except Exception as e:  # noqa: BLE001  (mamba_ssm missing on this machine)
        pytest.skip(str(e))
    cnn, fbank = hp["CNN"].eval(), hp["compute_features"]
    for n in [16000, 16001, 16159, 16160, 48000, 123457, 480000]:
        mel = fbank(torch.zeros(1, n))
        assert mel.shape[1] == student_mel_frames(n)
        with torch.no_grad():
            out = cnn(mel)
        assert out.shape[1] == cnn_out_frames(mel.shape[1]) == student_frames(n)
        assert out.shape[2] * out.shape[3] == 640


def test_cached_teacher_counts(work_dir):
    sf = pytest.importorskip("soundfile")
    sfx = os.environ.get("TEACHER_SUFFIX", "")
    idx_path = os.path.join(work_dir, "teacher", f"teacher_feats{sfx}.index.json")
    if not os.path.exists(idx_path):
        pytest.skip(f"{idx_path} not found")
    with open(idx_path) as f:
        index = json.load(f)["utts"]
    with open(os.path.join(work_dir, "data", "train_gt.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["ID"] in index]
    for r in random.Random(0).sample(rows, min(100, len(rows))):
        n = sf.info(r["wav"]).frames
        n_t = index[r["ID"]][1]
        assert n_t == teacher_frames(n), r["ID"]
        assert abs(student_frames(n) - n_t) <= 2, (r["ID"], student_frames(n), n_t)
