"""Sanity check 5: streaming path == full-sequence path.

Random weights (always, needs a GPU) and, if RUN_DIR is set, the trained
averaged model on 5 dev-clean utterances.
"""

import os

import pytest

from conftest import build_random_model, require_cuda_mamba


def test_streaming_random_weights(tmp_path):
    torch = require_cuda_mamba()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import streaming

    torch.manual_seed(0)
    hp = build_random_model(tmp_path)
    st = streaming.Student(hp, torch.zeros(80), torch.ones(80), "cuda")
    wavs = [torch.randn(1, 16000 * s) * 0.1 for s in (1, 3, 7)]
    ok, report = streaming.verify(st, None, wavs, ["r1", "r3", "r7"])
    assert ok, report


def test_streaming_cpu_matches_gpu(tmp_path):
    torch = require_cuda_mamba()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import streaming

    torch.manual_seed(0)
    hp = build_random_model(tmp_path)
    st = streaming.Student(hp, torch.zeros(80), torch.ones(80), "cuda")
    wav = torch.randn(1, 16000 * 2) * 0.1
    full = st.full_forward(st.mel(wav))[0].cpu()
    st.to("cpu")
    cpu = streaming.stream_utterance(st, st.mel(wav))[0]
    assert (cpu - full).abs().max().item() < 1e-3


def test_streaming_trained(run_dir):
    torch = require_cuda_mamba()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import streaming

    if not os.path.exists(os.path.join(run_dir, "final", "model_avg.ckpt")):
        pytest.skip("run has no exported final/model_avg.ckpt yet")
    st, hp = streaming.load_trained(run_dir, "avg", "cuda")
    sp = streaming.load_tokenizer(hp)
    rows = streaming.read_rows(hp, "dev-clean", 5)
    ok, report = streaming.verify(st, sp, [streaming.load_wav(r["wav"]) for r in rows],
                                  [r["ID"] for r in rows])
    assert ok, report
