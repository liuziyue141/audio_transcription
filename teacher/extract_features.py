#!/usr/bin/env python3
"""Teacher pass 2: Whisper-small final encoder features at 25 fps, fp16 memmap.

Writes (in --out_dir):
  teacher_feats.f16          all utterances back to back, float16, 768 columns
  teacher_feats.index.json   {"dim": 768, "fps": 25, "utts": {ID: [offset_frames, num_frames]},
                              "done": <rows completed>, "total_frames": ...}
  teacher_stats.npz          per-dimension mean / std over all training frames

Resumable: utterances are processed in a fixed order (CSV order); the index is
rewritten every --flush_every utterances and the binary file is truncated back
to the last recorded frame on restart.

Usage:
  python teacher/extract_features.py --train_csv /data/work/data/train_gt.csv \
      --out_dir /data/work/teacher [--limit 500]
  python teacher/extract_features.py ... --stats_only
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill_utils import teacher_enc_frames_50hz  # noqa: E402

DIM = 768


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_index(path, index):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f)
    os.replace(tmp, path)


def compute_stats(bin_path, total_frames, out_path, chunk=200_000):
    mm = np.memmap(bin_path, dtype=np.float16, mode="r", shape=(total_frames, DIM))
    s = np.zeros(DIM, np.float64)
    ss = np.zeros(DIM, np.float64)
    for i in range(0, total_frames, chunk):
        x = mm[i: i + chunk].astype(np.float64)
        s += x.sum(0)
        ss += (x * x).sum(0)
    mean = s / total_frames
    std = np.sqrt(np.maximum(ss / total_frames - mean ** 2, 1e-12))
    np.savez(out_path, mean=mean.astype(np.float32), std=std.astype(np.float32),
             frames=total_frames)
    print(f"[stats] frames={total_frames} mean|.|={np.abs(mean).mean():.4f} "
          f"std range=({std.min():.4f}, {std.max():.4f}) -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--flush_every", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stats_only", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sfx = f"_limit{args.limit}" if args.limit else ""
    bin_path = os.path.join(args.out_dir, f"teacher_feats{sfx}.f16")
    idx_path = os.path.join(args.out_dir, f"teacher_feats{sfx}.index.json")
    stats_path = os.path.join(args.out_dir, f"teacher_stats{sfx}.npz")

    rows = read_csv(args.train_csv)
    if args.limit:
        rows = rows[: args.limit]

    index = {"dim": DIM, "fps": 25, "model": args.model, "utts": {}, "done": 0, "total_frames": 0}
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            index = json.load(f)
        print(f"[resume] {index['done']} utterances done, {index['total_frames']} frames")
    # truncate any partially-written tail
    if os.path.exists(bin_path):
        with open(bin_path, "r+b") as f:
            f.truncate(index["total_frames"] * DIM * 2)

    if args.stats_only:
        compute_stats(bin_path, index["total_frames"], stats_path)
        return

    import soundfile as sf
    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel

    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    enc = WhisperModel.from_pretrained(args.model, torch_dtype=torch.float16).encoder
    enc = enc.cuda().eval()

    t0, audio_s, n_new = time.time(), 0.0, 0
    with open(bin_path, "ab") as fbin:
        start = index["done"]
        for b in range(start, len(rows), args.batch_size):
            batch = rows[b: b + args.batch_size]
            wavs = []
            for r in batch:
                a, sr = sf.read(r["wav"], dtype="float32")
                assert sr == 16000 and a.ndim == 1
                assert len(a) <= 30 * 16000, f"{r['ID']} longer than 30 s; filter train CSV first"
                wavs.append(a)
            inp = fe(wavs, sampling_rate=16000, return_tensors="pt").input_features
            with torch.no_grad():
                h = enc(inp.to("cuda", torch.float16)).last_hidden_state  # (B,1500,768), after final LN
            h = h.float().cpu().numpy()
            for r, a, feats in zip(batch, wavs, h):
                n_enc = teacher_enc_frames_50hz(len(a))
                feats = feats[:n_enc]
                n_t = n_enc // 2
                feats = feats[: 2 * n_t].reshape(n_t, 2, DIM).mean(1).astype(np.float16)
                fbin.write(feats.tobytes())
                index["utts"][r["ID"]] = [index["total_frames"], n_t]
                index["total_frames"] += n_t
                audio_s += len(a) / 16000
            index["done"] = b + len(batch)
            n_new += len(batch)
            if n_new // args.flush_every != (n_new - len(batch)) // args.flush_every:
                fbin.flush()
                os.fsync(fbin.fileno())
                save_index(idx_path, index)
                el = time.time() - t0
                print(f"[{index['done']}/{len(rows)}] RTF {el/audio_s:.5f} "
                      f"ETA {(len(rows)-index['done']) * el / n_new / 3600:.2f} h", flush=True)
        fbin.flush()
        os.fsync(fbin.fileno())
    save_index(idx_path, index)
    if n_new:
        el = time.time() - t0
        print(f"[timing] {n_new} utts in {el:.1f}s; extrapolated 28.5k utts: "
              f"{28539 * el / n_new / 3600:.2f} GPU h; size so far "
              f"{os.path.getsize(bin_path)/1e9:.2f} GB")
    compute_stats(bin_path, index["total_frames"], stats_path)


if __name__ == "__main__":
    main()
