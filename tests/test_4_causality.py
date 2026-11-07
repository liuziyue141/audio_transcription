"""Sanity check 4: changing mel frames >= k never changes earlier outputs.

Exact boundary (stricter than the plan's floor(k/4)-1 margin): student frame i
depends on mel frames up to 4i+6, so frames i < ceil((k-6)/4) must be bit-for-bit
stable (|diff| < 1e-5) and frame ceil((k-6)/4) must change.
"""

import pytest

from conftest import build_random_model, require_cuda_mamba
from distill_utils import first_affected_student_frame


def test_causality(tmp_path):
    torch = require_cuda_mamba()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    hp = build_random_model(tmp_path)
    cnn = hp["CNN"].cuda().eval()
    tr = hp["Transformer"].cuda().eval()
    lin = hp["ctc_lin"].cuda().eval()

    def run(x):
        with torch.no_grad():
            enc, _ = tr(cnn(x), tgt=None, wav_len=None)
            return torch.log_softmax(lin(enc), -1)

    x = torch.randn(1, 400, 80, device="cuda")
    y = run(x)
    for k in (100, 200, 300):
        x2 = x.clone()
        x2[:, k:] = torch.randn_like(x2[:, k:])
        y2 = run(x2)
        b = first_affected_student_frame(k)
        before = (y[:, :b] - y2[:, :b]).abs().max().item()
        at = (y[:, b] - y2[:, b]).abs().max().item()
        assert before < 1e-5, f"k={k}: frames < {b} changed by {before} -> future leak"
        assert at > 1e-5, f"k={k}: frame {b} did not change; receptive-field math is off"
