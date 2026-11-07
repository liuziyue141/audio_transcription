#!/usr/bin/env python3
"""Installation check 2: one causal ConmambaEncoder layer, forward + backward on GPU.

A (batch 8, length 1000, dim 256) input through one layer should take a few
milliseconds. Tens of ms or more means the fast mamba_ssm / causal-conv1d
CUDA kernels are not being used.

  python scripts/check_install.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def main():
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    assert torch.cuda.is_available(), "no CUDA GPU visible"
    import causal_conv1d  # noqa: F401
    import mamba_ssm
    import selective_scan_cuda  # noqa: F401  (compiled kernel of mamba-ssm)
    print(f"mamba_ssm {getattr(mamba_ssm, '__version__', '?')}, causal_conv1d OK, selective_scan_cuda OK")

    from modules.Conmamba import ConmambaEncoder

    cfg = {"d_state": 16, "expand": 2, "d_conv": 4, "bidirectional": False}
    enc = ConmambaEncoder(num_layers=1, d_model=256, d_ffn=1024, kernel_size=31,
                          activation=torch.nn.GELU, causal=True, mamba_config=cfg).cuda()
    x = torch.randn(8, 1000, 256, device="cuda", requires_grad=True)
    for _ in range(5):  # warm-up
        y, _ = enc(x)
        y.sum().backward()
    torch.cuda.synchronize()
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        y, _ = enc(x)
        y.sum().backward()
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0) / n
    print(f"1 causal ConMamba layer, (8,1000,256) fwd+bwd: {ms:.2f} ms")
    if ms > 30:
        print("WARNING: much slower than expected -> fast CUDA kernels probably not in use")
        sys.exit(1)
    print("INSTALL CHECK: PASS")


if __name__ == "__main__":
    main()
