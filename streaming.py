#!/usr/bin/env python3
"""Step-by-step (streaming) inference for the causal ConMamba CTC student.

Every step consumes 4 new mel frames (40 ms) and emits 1 student frame.
State kept between steps:
  * CNN front end: the mel frames still inside its receptive field
    (student frame i needs mel frames 4i .. 4i+6, see distill_utils.py)
  * per ConMamba layer: Mamba conv state + recurrent SSM state, and the causal
    ConvolutionModule's last (kernel_size - 1) frames
Feed-forward / LayerNorm parts act frame by frame and keep no state.

The Mamba step is pure PyTorch (works on CPU and GPU) and mirrors
mamba_ssm 1.1.x `Mamba.step`.

Subcommands (run_dir = a finished training run with final/model_avg.ckpt):
  python streaming.py verify  --run_dir RUN [--n 5]                # sanity check 5
  python streaming.py bench   --run_dir RUN [--n_rtf 200 --n_stream 20]
  python streaming.py emissions --run_dir RUN [--split dev-clean]  # token frame indices
Add --random_init to run verify without a trained model (equivalence holds for any weights).
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_utils import MEL_PER_STUDENT_FRAME, RF_RIGHT  # noqa: E402

CNN_WINDOW = RF_RIGHT + 1  # 7 mel frames -> exactly 1 student frame


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_hparams(hparams_file, overrides=None):
    from hyperpyyaml import load_hyperpyyaml

    with open(hparams_file) as f:
        return load_hyperpyyaml(f, overrides or {})


def load_trained(run_dir, which="avg", device="cpu"):
    """Load YAML saved in the run dir + exported averaged (or best) weights."""
    hp = load_hparams(os.path.join(run_dir, "hyperparams.yaml"))
    sd = torch.load(os.path.join(run_dir, "final", f"model_{which}.ckpt"), map_location="cpu")
    hp["model"].load_state_dict(sd, strict=True)
    norm = hp["normalize"]
    norm._load(os.path.join(run_dir, "final", "normalizer.ckpt"))
    mean, std = norm.glob_mean.detach().float().cpu(), norm.glob_std.detach().float().cpu()
    return Student(hp, mean, std, device), hp


class Student:
    """Wraps the trained parts; provides full-sequence and streaming forward."""

    def __init__(self, hp, mean, std, device="cpu"):
        self.hp = hp
        self.device = torch.device(device)
        self.fbank = hp["compute_features"].to(self.device)
        self.cnn = hp["CNN"].to(self.device).eval()
        self.tr = hp["Transformer"].to(self.device).eval()
        self.ctc_lin = hp["ctc_lin"].to(self.device).eval()
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        enc = self.tr.encoder
        assert type(enc).__name__ == "ConmambaEncoder"
        for layer in enc.layers:
            assert not hasattr(layer.mamba, "bimamba_type"), "bidirectional Mamba found"
            assert layer.convolution_module.causal and layer.convolution_module.dilation == 1

    def to(self, device):
        self.device = torch.device(device)
        for m in (self.fbank, self.cnn, self.tr, self.ctc_lin):
            m.to(self.device)
        self.mean, self.std = self.mean.to(self.device), self.std.to(self.device)
        return self

    def num_params(self):
        return sum(p.numel() for p in self.hp["model"].parameters())

    @torch.no_grad()
    def mel(self, wav):
        """wav (B, N) on CPU or device -> normalized mel (B, T, 80) on device."""
        feats = self.fbank(wav.to(self.device))
        return (feats - self.mean) / self.std

    @torch.no_grad()
    def full_forward(self, mel):
        src = self.cnn(mel)
        enc, _ = self.tr(src, tgt=None, wav_len=None)
        return F.log_softmax(self.ctc_lin(enc), dim=-1)

    def streamer(self):
        return StreamState(self)


# --------------------------------------------------------------------------
# Streaming state
# --------------------------------------------------------------------------
def mamba_step(m, h, st):
    """One time step of mamba_ssm.Mamba (pure PyTorch). h: (B, 1, D)."""
    xz = m.in_proj(h.squeeze(1))
    x, z = xz.chunk(2, dim=-1)
    conv = torch.roll(st["conv"], shifts=-1, dims=-1)
    conv[:, :, -1] = x
    st["conv"] = conv
    x = (conv * m.conv1d.weight.squeeze(1)).sum(-1)
    if m.conv1d.bias is not None:
        x = x + m.conv1d.bias
    x = F.silu(x)
    x_db = m.x_proj(x)
    dt, B, C = torch.split(x_db, [m.dt_rank, m.d_state, m.d_state], dim=-1)
    dt = F.softplus(F.linear(dt, m.dt_proj.weight) + m.dt_proj.bias)
    A = -torch.exp(m.A_log.float())
    dA = torch.exp(dt.unsqueeze(-1) * A)
    dB = dt.unsqueeze(-1) * B.unsqueeze(1)
    st["ssm"] = st["ssm"] * dA + x.unsqueeze(-1) * dB
    y = (st["ssm"] * C.unsqueeze(1)).sum(-1) + m.D * x
    y = y * F.silu(z)
    return m.out_proj(y).unsqueeze(1)


def conv_module_step(cm, x, st):
    """One step of the causal ConvolutionModule. x: (B, 1, D)."""
    out = cm.layer_norm(x).transpose(1, 2)        # (B, D, 1)
    out = cm.bottleneck(out)                       # pointwise conv + GLU
    buf = torch.cat([st["cm"], out], dim=2)        # (B, D, k)
    st["cm"] = buf[:, :, 1:]
    y = F.conv1d(buf, cm.conv.weight, cm.conv.bias, padding=0, groups=cm.conv.groups)
    return cm.after_conv(y.transpose(1, 2))


class StreamState:
    def __init__(self, student, batch=1):
        self.s = student
        dev = student.device
        self.mel_buf = torch.zeros(batch, 0, 80, device=dev)
        self.buf_start = 0   # mel index of mel_buf[:, 0]
        self.next_i = 0      # next student frame to emit
        self.layers = []
        for layer in student.tr.encoder.layers:
            m, cm = layer.mamba, layer.convolution_module
            self.layers.append({
                "conv": torch.zeros(batch, m.d_inner, m.d_conv, device=dev),
                "ssm": torch.zeros(batch, m.d_inner, m.d_state, device=dev),
                "cm": torch.zeros(batch, cm.conv.in_channels, cm.kernel_size - 1, device=dev),
            })

    @torch.no_grad()
    def _emit_one(self):
        s = self.s
        lo = MEL_PER_STUDENT_FRAME * self.next_i - self.buf_start
        window = self.mel_buf[:, lo: lo + CNN_WINDOW]
        x = s.cnn(window)                                  # (B, 1, 20, 32)
        x = x.reshape(x.shape[0], x.shape[1], -1)          # (B, 1, 640)
        x = s.tr.custom_src_module(x)
        for layer, st in zip(s.tr.encoder.layers, self.layers):
            x = x + 0.5 * layer.ffn_module1(x)
            x = mamba_step(layer.mamba, layer.norm1(x), st) + x
            x = x + conv_module_step(layer.convolution_module, x, st)
            x = layer.norm2(x + 0.5 * layer.ffn_module2(x))
        x = s.tr.encoder.norm(x)
        self.next_i += 1
        return F.log_softmax(s.ctc_lin(x), dim=-1)          # (B, 1, V)

    @torch.no_grad()
    def push(self, mel_chunk):
        """Append normalized mel frames (B, n, 80); return new log-probs (B, k, V)."""
        self.mel_buf = torch.cat([self.mel_buf, mel_chunk.to(self.mel_buf.device)], dim=1)
        outs = []
        while (self.buf_start + self.mel_buf.shape[1]
               > MEL_PER_STUDENT_FRAME * self.next_i + RF_RIGHT):
            outs.append(self._emit_one())
        drop = MEL_PER_STUDENT_FRAME * self.next_i - self.buf_start
        if drop > 0:
            self.mel_buf = self.mel_buf[:, drop:]
            self.buf_start += drop
        if outs:
            return torch.cat(outs, dim=1)
        return self.mel_buf.new_zeros(self.mel_buf.shape[0], 0, self.s.ctc_lin.w.out_features)


def stream_utterance(student, mel, step_frames=MEL_PER_STUDENT_FRAME, timer=None):
    st = student.streamer()
    outs = []
    for t in range(0, mel.shape[1], step_frames):
        if timer is not None:
            timer.start()
        o = st.push(mel[:, t: t + step_frames])
        if timer is not None:
            timer.stop(o.shape[1])
        outs.append(o)
    return torch.cat(outs, dim=1)


# --------------------------------------------------------------------------
# Decoding helpers
# --------------------------------------------------------------------------
def greedy_with_frames(logp, blank=0):
    """(T, V) log-probs -> (token ids, frame index of each emitted token)."""
    best = logp.argmax(-1).tolist()
    toks, frames, prev = [], [], None
    for t, k in enumerate(best):
        if k != blank and k != prev:
            toks.append(k)
            frames.append(t)
        prev = k
    return toks, frames


def load_tokenizer(hp):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(hp["tokenizer_folder"],
                         f"{hp['output_neurons']}_{hp['token_type']}.model"))
    return sp


def read_rows(hp, split, n=None):
    path = os.path.join(hp["data_csv_folder"], f"{split}.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = sorted(rows, key=lambda r: r["ID"])
    return rows[:n] if n else rows


def load_wav(path):
    import soundfile as sf

    a, sr = sf.read(path, dtype="float32")
    assert sr == 16000
    return torch.from_numpy(a).unsqueeze(0)


class Timer:
    def __init__(self, cuda):
        self.cuda, self.total, self.frames, self.calls, self._t = cuda, 0.0, 0, 0, None

    def start(self):
        if self.cuda:
            torch.cuda.synchronize()
        self._t = time.perf_counter()

    def stop(self, n_frames):
        if self.cuda:
            torch.cuda.synchronize()
        if n_frames:
            self.total += time.perf_counter() - self._t
            self.frames += n_frames
            self.calls += 1

    def ms_per_frame(self):
        return 1000.0 * self.total / max(1, self.frames)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def verify(student, sp, wavs, ids, atol=1e-3):
    """Sanity check 5: streaming == full-sequence."""
    ok = True
    report = []
    for uid, wav in zip(ids, wavs):
        mel = student.mel(wav)
        full = student.full_forward(mel)[0]
        strm = stream_utterance(student, mel)[0]
        assert strm.shape == full.shape, f"{uid}: stream {tuple(strm.shape)} vs full {tuple(full.shape)}"
        diff = (strm - full).abs().max().item()
        tf, _ = greedy_with_frames(full)
        ts, _ = greedy_with_frames(strm)
        same = tf == ts
        if sp is not None:
            same_txt = sp.decode_ids(tf) == sp.decode_ids(ts)
        else:
            same_txt = same
        passed = diff < atol and same and same_txt
        ok &= passed
        report.append({"id": uid, "frames": full.shape[0], "max_abs_diff": diff,
                       "same_greedy": same, "pass": passed})
        print(f"{uid}: frames={full.shape[0]} max|diff|={diff:.2e} same_greedy={same} "
              f"{'PASS' if passed else 'FAIL'}", flush=True)
    return ok, report


def bench(student, rows, n_rtf, n_stream, device):
    res = {"params_M": student.num_params() / 1e6,
           "fp32_MB": student.num_params() * 4 / 2 ** 20}
    cuda = device.startswith("cuda")
    if cuda:
        # offline RTF, full sequence, batch 1
        wavs = [load_wav(r["wav"]) for r in rows[:n_rtf]]
        for w in wavs[:3]:
            student.full_forward(student.mel(w))
        torch.cuda.synchronize()
        t0, audio = time.perf_counter(), 0.0
        for w in wavs:
            student.full_forward(student.mel(w))
            audio += w.shape[1] / 16000
        torch.cuda.synchronize()
        res["gpu_offline_rtf"] = (time.perf_counter() - t0) / audio
        # streaming step time on GPU
        tm = Timer(True)
        for r in rows[:n_stream]:
            stream_utterance(student, student.mel(load_wav(r["wav"])), timer=tm)
        res["gpu_stream_ms_per_40ms_frame"] = tm.ms_per_frame()
    # streaming on one CPU thread
    torch.set_num_threads(1)
    cpu_student = student.to("cpu")
    tm = Timer(False)
    for r in rows[:n_stream]:
        stream_utterance(cpu_student, cpu_student.mel(load_wav(r["wav"])), timer=tm)
    res["cpu1_stream_ms_per_40ms_frame"] = tm.ms_per_frame()
    res["cpu1_stream_rtf"] = res["cpu1_stream_ms_per_40ms_frame"] / 40.0
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["verify", "bench", "emissions"])
    p.add_argument("--run_dir", help="training output folder (has hyperparams.yaml, final/)")
    p.add_argument("--which", default="avg", choices=["avg", "best"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split", default="dev-clean")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--n_rtf", type=int, default=200)
    p.add_argument("--n_stream", type=int, default=20)
    p.add_argument("--random_init", action="store_true",
                   help="verify with random weights: --hparams YAML --work_dir needed")
    p.add_argument("--hparams", default="hparams/CTC/conmamba_causal_distill.yaml")
    p.add_argument("--work_dir")
    p.add_argument("--out")
    args = p.parse_args()

    # exact fp32 comparisons: no TF32 shortcuts on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if args.random_init:
        torch.manual_seed(0)
        hp = load_hparams(args.hparams, {"work_dir": args.work_dir or "/tmp/unused",
                                         "data_folder": "unused"})
        student = Student(hp, torch.zeros(80), torch.ones(80), args.device)
        sp = None
    else:
        student, hp = load_trained(args.run_dir, args.which, args.device)
        sp = load_tokenizer(hp)

    if args.cmd == "verify":
        if args.random_init and not args.work_dir:
            ids = [f"random{i}" for i in range(args.n)]
            wavs = [torch.randn(1, 16000 * (3 + 2 * i)) * 0.1 for i in range(args.n)]
        else:
            rows = read_rows(hp, args.split, args.n)
            ids, wavs = [r["ID"] for r in rows], [load_wav(r["wav"]) for r in rows]
        ok, report = verify(student, sp, wavs, ids)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2)
        print("STREAMING EQUIVALENCE:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    if args.cmd == "bench":
        rows = read_rows(hp, args.split)
        res = bench(student, rows, args.n_rtf, args.n_stream, args.device)
        out = args.out or os.path.join(args.run_dir, "final", f"speed_{args.which}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))

    if args.cmd == "emissions":
        rows = read_rows(hp, args.split)
        out = args.out or os.path.join(args.run_dir, "final", f"emissions_{args.which}_{args.split}.jsonl")
        with open(out, "w") as f:
            for r in rows:
                logp = stream_utterance(student, student.mel(load_wav(r["wav"])))[0]
                toks, frames = greedy_with_frames(logp)
                f.write(json.dumps({"ID": r["ID"], "hyp": sp.decode_ids(toks),
                                    "tokens": [sp.id_to_piece(t) for t in toks],
                                    "frames": frames, "frame_ms": 40,
                                    "note": "frame i needs audio up to mel frame 4i+6"}) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
