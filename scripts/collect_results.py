#!/usr/bin/env python3
"""Collect WER / speed tables and training curves for the results report.

  python scripts/collect_results.py --work_dir /data/work \
      --runs E0_20261106 E2_20261106 E1_20261106 E3_20261106
Writes <work_dir>/report/results.md and curves.png.
"""

import argparse
import json
import os
import re

SETS = ["dev-clean", "dev-other", "test-clean", "test-other"]


def load_json(p):
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def valid_curve(train_log):
    out = []
    if not os.path.exists(train_log):
        return out
    for line in open(train_log):
        e = re.search(r"epoch: (\d+)", line)
        w = re.search(r"valid WER: ([\d.e+-]+)", line)
        if e and w:
            out.append((int(e.group(1)), float(w.group(1))))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args()
    rep = os.path.join(args.work_dir, "report")
    os.makedirs(rep, exist_ok=True)

    lines = ["## WER (%) — greedy CTC, no LM, scored against ground truth", "",
             "| model | " + " | ".join(SETS) + " |", "|---" * (len(SETS) + 1) + "|"]
    for run in args.runs:
        res = load_json(os.path.join(args.work_dir, "runs", run, "results_greedy.json")) or {}
        for mode in ("average", "best"):
            r = res.get(mode, {})
            cells = [f"{r[s]['WER']:.2f}" if s in r else "—" for s in SETS]
            lines.append(f"| {run} ({mode}) | " + " | ".join(cells) + " |")
    for m in ("small", "large-v3"):
        r = load_json(os.path.join(args.work_dir, "whisper_ref", f"{m}_wer.json"))
        if r:
            lines.append(f"| Whisper {m} (offline) | " + " | ".join(f"{r[s]:.2f}" for s in SETS) + " |")

    lines += ["", "## Speed", "",
              "| model | params (M) | fp32 MB | GPU offline RTF | GPU stream ms/40ms | CPU-1thr stream ms/40ms |",
              "|---|---|---|---|---|---|"]
    for run in args.runs:
        s = load_json(os.path.join(args.work_dir, "runs", run, "final", "speed_avg.json"))
        if s:
            lines.append(f"| {run} | {s['params_M']:.2f} | {s['fp32_MB']:.1f} | "
                         f"{s.get('gpu_offline_rtf', float('nan')):.4f} | "
                         f"{s.get('gpu_stream_ms_per_40ms_frame', float('nan')):.2f} | "
                         f"{s['cpu1_stream_ms_per_40ms_frame']:.2f} |")
    pl = load_json(os.path.join(args.work_dir, "data", "pl_report.json"))
    if pl:
        lines += ["", "## Pseudo-labels", "", "```", json.dumps(pl, indent=2), "```"]
    with open(os.path.join(rep, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping curves")
        return
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for run in args.runs:
        sl = os.path.join(args.work_dir, "runs", run, "step_log.jsonl")
        if os.path.exists(sl):
            recs = [json.loads(x) for x in open(sl) if x.strip()]
            st = [r["step"] for r in recs]
            ax[0, 0].plot(st, [r["loss_ctc"] for r in recs], label=run)
            if "loss_distill" in recs[-1]:
                ax[0, 1].plot(st, [r.get("loss_distill") for r in recs], label=run)
                ax[1, 0].plot(st, [r.get("cos") for r in recs], label=run)
        vc = valid_curve(os.path.join(args.work_dir, "runs", run, "train_log.txt"))
        if vc:
            ax[1, 1].plot(*zip(*vc), label=run)
    for a, t in zip(ax.flat, ["CTC loss (step)", "distill loss (step)",
                              "student-teacher cosine (step)", "dev-clean WER (epoch)"]):
        a.set_title(t)
        a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(rep, "curves.png"), dpi=120)
    print(f"saved {os.path.join(rep, 'curves.png')}")


if __name__ == "__main__":
    main()
