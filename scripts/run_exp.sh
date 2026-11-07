#!/usr/bin/env bash
# Launch one baseline experiment in the background (survives SSH drops).
#
#   RUN_NAME=E0_20261106 bash scripts/run_exp.sh E0 [extra --key value overrides]
#
# To resume after a spot interruption, run the SAME command again (same RUN_NAME):
# Brain.fit() recovers from the newest checkpoint in <work_dir>/runs/<RUN_NAME>/save.
set -euo pipefail
EXP="${1:?usage: RUN_NAME=... $0 E0|E1|E2|E3|E0-sub [overrides]}"; shift
W="${WORK_DIR:-/data/work}"
D="${DATA_FOLDER:-/data/LibriSpeech}"
NAME="${RUN_NAME:?set RUN_NAME, e.g. RUN_NAME=${EXP}_$(date +%Y%m%d)}"
case "$EXP" in
  E0)     CSV=train_gt.csv;        KD=0.0 ;;
  E2)     CSV=train_gt.csv;        KD=1.0 ;;
  E1)     CSV=train_pl.csv;        KD=0.0 ;;
  E3)     CSV=train_pl.csv;        KD=1.0 ;;
  E0-sub) CSV=train_gt_plkept.csv; KD=0.0 ;;
  *) echo "unknown experiment $EXP"; exit 1 ;;
esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$W/logs"
cd "$REPO"
nohup python train_distill_CTC.py hparams/CTC/conmamba_causal_distill.yaml \
  --work_dir "$W" --data_folder "$D" --experiment "$NAME" \
  --train_csv "$W/data/$CSV" --kd_weight "$KD" \
  --ckpt_interval_minutes 10 \
  ${S3_URI:+--s3_uri "$S3_URI/$NAME"} \
  "$@" >> "$W/logs/$NAME.log" 2>&1 &
echo "started $NAME (pid $!)  log: $W/logs/$NAME.log"
