#!/usr/bin/env python3
"""Data preparation for the Whisper -> causal ConMamba baseline (training env).

Builds, once, in <work_dir>/data:
  train-clean-100.csv, dev-clean.csv, dev-other.csv, test-clean.csv, test-other.csv
  train_gt.csv            ground truth, rows > 30.0 s removed (never edited later)
  train_gt_10h.csv        speaker subset of ~10 h, for sanity check 10
  train_gt_overfit100.csv 100 utterances, for sanity checks 6 and 7
and trains the shared character tokenizer in <work_dir>/tokenizer.

Usage:
  python prepare_distill_data.py --data_folder /data/LibriSpeech --work_dir /data/work
"""

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text_norm import is_librispeech_style, normalize  # noqa: E402

TRAIN_SPLITS = ["train-clean-100"]
EVAL_SPLITS = ["dev-clean", "dev-other", "test-clean", "test-other"]
MAX_SECONDS = 30.0


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def hours(rows):
    return sum(float(r["duration"]) for r in rows) / 3600.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_folder", required=True, help=".../LibriSpeech")
    p.add_argument("--work_dir", required=True)
    p.add_argument("--subset_hours", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=3402)
    args = p.parse_args()

    from librispeech_prepare import prepare_librispeech

    data_dir = os.path.join(args.work_dir, "data")
    tok_dir = os.path.join(args.work_dir, "tokenizer")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    prepare_librispeech(
        data_folder=args.data_folder,
        save_folder=data_dir,
        tr_splits=TRAIN_SPLITS,
        dev_splits=[],
        te_splits=EVAL_SPLITS,
        merge_lst=[],
        merge_name=None,
        skip_prep=False,
    )

    stats = {}
    raw = read_csv(os.path.join(data_dir, "train-clean-100.csv"))
    fields = list(raw[0].keys())

    # Normalizer identity assertion on every ground-truth row
    bad = [r["ID"] for r in raw if normalize(r["wrd"]) != r["wrd"]]
    if bad:
        raise SystemExit(f"normalize() changed {len(bad)} GT rows, e.g. {bad[:5]}: normalizer bug")
    assert all(is_librispeech_style(r["wrd"]) for r in raw)

    kept = [r for r in raw if float(r["duration"]) <= MAX_SECONDS]
    removed = len(raw) - len(kept)
    print(f"[filter] removed {removed} train rows longer than {MAX_SECONDS}s")
    write_csv(os.path.join(data_dir, "train_gt.csv"), kept, fields)
    stats["train_gt"] = {"utts": len(kept), "hours": round(hours(kept), 3), "removed_gt_30s": removed}

    for split in EVAL_SPLITS:
        rows = read_csv(os.path.join(data_dir, f"{split}.csv"))
        stats[split] = {"utts": len(rows), "hours": round(hours(rows), 3)}

    # ~10 h speaker subset (whole speakers, deterministic)
    rng = random.Random(args.seed)
    by_spk = {}
    for r in kept:
        by_spk.setdefault(r["spk_id"], []).append(r)
    spks = sorted(by_spk)
    rng.shuffle(spks)
    subset, total = [], 0.0
    for s in spks:
        if total >= args.subset_hours * 3600:
            break
        subset.extend(by_spk[s])
        total += sum(float(r["duration"]) for r in by_spk[s])
    write_csv(os.path.join(data_dir, "train_gt_10h.csv"), subset, fields)
    stats["train_gt_10h"] = {"utts": len(subset), "hours": round(hours(subset), 3), "speakers": len({r['spk_id'] for r in subset})}

    # 100 utterances for overfitting, short-ish ones so it is fast
    pool = sorted([r for r in kept if 4.0 <= float(r["duration"]) <= 12.0], key=lambda r: r["ID"])
    over = rng.sample(pool, 100)
    write_csv(os.path.join(data_dir, "train_gt_overfit100.csv"), over, fields)
    stats["train_gt_overfit100"] = {"utts": len(over), "hours": round(hours(over), 3)}

    # Shared tokenizer, trained once from train_gt.csv
    from speechbrain.tokenizers.SentencePiece import SentencePiece

    tok = SentencePiece(
        model_dir=tok_dir,
        vocab_size=31,
        annotation_train=os.path.join(data_dir, "train_gt.csv"),
        annotation_read="wrd",
        model_type="char",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
    )
    n = tok.sp.vocab_size()
    assert n == 31, f"tokenizer has {n} entries, expected 31"
    stats["tokenizer"] = {"dir": tok_dir, "vocab_size": n,
                          "pieces": [tok.sp.id_to_piece(i) for i in range(n)]}

    with open(os.path.join(data_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
