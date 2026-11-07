#!/usr/bin/env python3
"""Offline Whisper reference WER on dev/test sets (teacher env).

Usage:
  python teacher/eval_whisper.py --data_dir /data/work/data --model small --out_dir /data/work/whisper_ref
  python teacher/eval_whisper.py --data_dir /data/work/data --model large-v3 --out_dir /data/work/whisper_ref
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from make_pseudo_labels import load_whisper, transcribe  # noqa: E402
from text_norm import normalize  # noqa: E402

SPLITS = ["dev-clean", "dev-other", "test-clean", "test-other"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model", required=True, help="faster-whisper name: small, large-v3, ...")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    import jiwer

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_whisper(args.model)
    results = {}
    for split in SPLITS:
        with open(os.path.join(args.data_dir, f"{split}.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        if args.limit:
            rows = rows[: args.limit]
        refs, hyps = [], []
        out_path = os.path.join(args.out_dir, f"{args.model}_{split}.jsonl")
        with open(out_path, "w") as fo:
            for r in rows:
                raw, _, _ = transcribe(model, r["wav"])
                hyp = normalize(raw)
                refs.append(r["wrd"])
                hyps.append(hyp if hyp else "<EMPTY>")
                fo.write(json.dumps({"ID": r["ID"], "ref": r["wrd"], "hyp": hyp, "raw": raw}) + "\n")
        results[split] = round(100 * jiwer.wer(refs, hyps), 3)
        print(f"{args.model} {split}: WER {results[split]:.2f}%", flush=True)
    with open(os.path.join(args.out_dir, f"{args.model}_wer.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
