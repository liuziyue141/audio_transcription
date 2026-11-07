#!/usr/bin/env python3
"""Teacher pass 1: Whisper large-v3 pseudo-labels (teacher env, faster-whisper).

Reads train_gt.csv, writes (in --out_dir):
  train_pl_raw.jsonl   one line per utterance as it is transcribed (resumable)
  train_pl_meta.csv    ID, raw text, normalized text, avg_logprob, no_speech_prob, wer, keep
  train_pl.csv         like train_gt.csv but wrd = pseudo-label, kept rows only (E1, E3)
  train_gt_plkept.csv  ground truth on the same kept rows (E0-sub)
  pl_report.json       counts, hours, corpus WER of pseudo-labels vs ground truth
  pl_dropped_sample.txt  20 random dropped examples for manual inspection

Usage:
  python teacher/make_pseudo_labels.py --train_csv /data/work/data/train_gt.csv \
      --out_dir /data/work/data [--limit 500]
"""

import argparse
import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_norm import normalize  # noqa: E402

WER_THRESHOLD = 0.20
MAX_REPEAT = 4


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_whisper(model_name="large-v3", device="cuda", compute_type="float16"):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe(model, wav_path, beam_size=5):
    """Return (raw_text, avg_logprob, no_speech_prob) for one utterance."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000, f"{wav_path}: expected 16 kHz, got {sr}"
    segments, _info = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        beam_size=beam_size,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
        vad_filter=False,
    )
    segments = list(segments)
    text = " ".join(s.text.strip() for s in segments).strip()
    if segments:
        avg_lp = sum(s.avg_logprob for s in segments) / len(segments)
        nsp = max(s.no_speech_prob for s in segments)
    else:
        avg_lp, nsp = float("-inf"), 1.0
    return text, avg_lp, nsp


def max_word_repeat(text):
    words, best, run = text.split(), 0, 0
    for i, w in enumerate(words):
        run = run + 1 if i > 0 and w == words[i - 1] else 1
        best = max(best, run)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="large-v3")
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N rows (timing run); writes *_limitN files")
    p.add_argument("--seed", type=int, default=3402)
    args = p.parse_args()

    import jiwer

    rows = read_csv(args.train_csv)
    fields = list(rows[0].keys())
    suffix = ""
    if args.limit:
        rows = rows[: args.limit]
        suffix = f"_limit{args.limit}"
    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, f"train_pl_raw{suffix}.jsonl")

    done = {}
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    done[d["ID"]] = d
        print(f"[resume] {len(done)} utterances already transcribed")

    model = load_whisper(args.model)
    t0, n_new, audio_s = time.time(), 0, 0.0
    with open(raw_path, "a") as fout:
        for i, r in enumerate(rows):
            if r["ID"] in done:
                continue
            text, lp, nsp = transcribe(model, r["wav"], args.beam_size)
            d = {"ID": r["ID"], "raw": text, "avg_logprob": lp, "no_speech_prob": nsp}
            fout.write(json.dumps(d) + "\n")
            done[r["ID"]] = d
            n_new += 1
            audio_s += float(r["duration"])
            if n_new % 200 == 0:
                fout.flush()
                el = time.time() - t0
                rem = len(rows) - len(done)
                print(f"[{len(done)}/{len(rows)}] {el/n_new:.3f}s/utt, RTF {el/audio_s:.4f}, "
                      f"ETA {rem * el / n_new / 3600:.2f} h", flush=True)
    if n_new:
        el = time.time() - t0
        print(f"[timing] {n_new} utts in {el:.1f}s = {el/n_new:.3f}s/utt, RTF {el/audio_s:.4f}")
        print(f"[timing] extrapolated full train set (28.5k utts): {28539 * el / n_new / 3600:.2f} GPU h")

    # ---- normalize, score, filter
    meta, kept_pl, kept_gt, dropped = [], [], [], []
    refs, hyps = [], []
    for r in rows:
        d = done[r["ID"]]
        hyp = normalize(d["raw"])
        ref = r["wrd"]
        w = jiwer.wer(ref, hyp) if hyp else 1.0
        rep = max_word_repeat(hyp)
        keep = bool(hyp) and w <= WER_THRESHOLD and rep <= MAX_REPEAT
        reason = "" if keep else ("empty" if not hyp else ("repeat" if rep > MAX_REPEAT else "wer"))
        meta.append({"ID": r["ID"], "raw": d["raw"], "normalized": hyp,
                     "avg_logprob": d["avg_logprob"], "no_speech_prob": d["no_speech_prob"],
                     "wer": round(w, 4), "keep": int(keep), "drop_reason": reason})
        refs.append(ref)
        hyps.append(hyp if hyp else "<EMPTY>")
        if keep:
            kept_pl.append({**r, "wrd": hyp})
            kept_gt.append(r)
        else:
            dropped.append((r, hyp, w, reason))

    write_csv(os.path.join(args.out_dir, f"train_pl_meta{suffix}.csv"), meta, list(meta[0].keys()))
    write_csv(os.path.join(args.out_dir, f"train_pl{suffix}.csv"), kept_pl, fields)
    write_csv(os.path.join(args.out_dir, f"train_gt_plkept{suffix}.csv"), kept_gt, fields)

    hrs = lambda rs: sum(float(x["duration"]) for x in rs) / 3600
    total_h = hrs(rows)
    report = {
        "utts_total": len(rows), "utts_kept": len(kept_pl), "utts_dropped": len(dropped),
        "hours_total": round(total_h, 3), "hours_kept": round(hrs(kept_pl), 3),
        "hours_dropped_pct": round(100 * (1 - hrs(kept_pl) / total_h), 3),
        "corpus_wer_all_pct": round(100 * jiwer.wer(refs, hyps), 3),
        "corpus_wer_kept_pct": round(100 * jiwer.wer([r["wrd"] for r in kept_gt],
                                                     [r["wrd"] for r in kept_pl]), 3) if kept_pl else None,
        "drop_reasons": {k: sum(1 for x in dropped if x[3] == k) for k in ("empty", "repeat", "wer")},
        "run_E0_sub": hrs(kept_pl) < 0.95 * total_h,
    }
    with open(os.path.join(args.out_dir, f"pl_report{suffix}.json"), "w") as f:
        json.dump(report, f, indent=2)
    rng = random.Random(args.seed)
    sample = rng.sample(dropped, min(20, len(dropped)))
    with open(os.path.join(args.out_dir, f"pl_dropped_sample{suffix}.txt"), "w") as f:
        for r, hyp, w, reason in sample:
            f.write(f"{r['ID']}  wer={w:.3f}  reason={reason}\n  REF: {r['wrd']}\n"
                    f"  HYP: {hyp}\n  RAW: {done[r['ID']]['raw']}\n\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
