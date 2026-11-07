"""Sanity check 11: checkpoint averaging excludes distill_proj and intra-epoch ckpts.

Run after check 9 (or any run) with RUN_DIR pointing at its output folder.
"""

import os

import pytest


def test_average_excludes_training_only_parts(run_dir, tmp_path):
    pytest.importorskip("speechbrain")
    import speechbrain as sb
    from hyperpyyaml import load_hyperpyyaml

    try:
        from speechbrain.core import INTRA_EPOCH_CKPT_FLAG
    except ImportError:
        INTRA_EPOCH_CKPT_FLAG = "brain_intra_epoch_ckpt"

    with open(os.path.join(run_dir, "hyperparams.yaml")) as f:
        hp = load_hyperpyyaml(f)
    ckpter = hp["checkpointer"]
    all_ckpts = ckpter.list_checkpoints()
    assert all_ckpts, "no checkpoints in save folder"
    ckpts = ckpter.find_checkpoints(
        min_key="WER",
        ckpt_predicate=lambda c: INTRA_EPOCH_CKPT_FLAG not in c.meta and "WER" in c.meta)
    if not ckpts:
        pytest.skip("only intra-epoch checkpoints so far (no finished epoch)")
    assert all(INTRA_EPOCH_CKPT_FLAG not in c.meta for c in ckpts)
    avg = sb.utils.checkpoints.average_checkpoints(ckpts, recoverable_name="model")
    assert not [k for k in avg if "distill_proj" in k or "teacher" in k]
    hp["model"].load_state_dict(avg, strict=True)
    # distill_proj is still resumable on its own
    assert all("distill_proj" in c.paramfiles for c in ckpts)
