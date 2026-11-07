#!/usr/bin/env python3
"""Sanity check 8: teacher feature L2 norm over time vs the waveform.

Usage:
  python teacher/plot_features.py --train_csv /data/work/data/train_gt.csv \
      --teacher_dir /data/work/teacher --out /data/work/teacher/check8.png
"""

import argparse
import csv
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--teacher_dir", required=True)
    p.add_argument("--suffix", default="", help="e.g. _limit500")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    with open(os.path.join(args.teacher_dir, f"teacher_feats{args.suffix}.index.json")) as f:
        index = json.load(f)
    mm = np.memmap(os.path.join(args.teacher_dir, f"teacher_feats{args.suffix}.f16"),
                   dtype=np.float16, mode="r", shape=(index["total_frames"], index["dim"]))
    with open(args.train_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["ID"] in index["utts"]]
    rows = sorted(rows, key=lambda r: float(r["duration"]))
    picks = [rows[len(rows) // 4], rows[len(rows) // 2], rows[3 * len(rows) // 4]][: args.n]

    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 3 * len(picks)))
    for ax, r in zip(np.atleast_1d(axes), picks):
        a, sr = sf.read(r["wav"], dtype="float32")
        off, n = index["utts"][r["ID"]]
        norm = np.linalg.norm(mm[off: off + n].astype(np.float32), axis=1)
        t_wav = np.arange(len(a)) / sr
        t_feat = (np.arange(n) + 0.5) * 0.04
        ax.plot(t_wav, a / (np.abs(a).max() + 1e-9), color="0.7", lw=0.5, label="waveform")
        ax2 = ax.twinx()
        ax2.plot(t_feat, norm, color="C3", lw=1.2, label="teacher ||h||")
        ax.set_title(f"{r['ID']}  ({r['duration']} s, {n} teacher frames)")
        ax.set_xlim(0, len(a) / sr)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
