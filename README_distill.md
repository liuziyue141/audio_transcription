# Whisper → streaming ConMamba (teacher–student baseline): how to run

Everything runs on **one AWS `g5.xlarge`** (1× NVIDIA A10G 24 GB, 4 vCPU, 16 GB RAM),
spot pricing. A Linux CUDA GPU is required: the Mamba kernels (`mamba-ssm`,
`causal-conv1d`) do not run on a Mac. The A10G is an Ampere GPU (sm_86) and
supports bf16, which the config uses.

All paths below assume:

| what | path |
|---|---|
| EBS data volume | `/data` (200 GB gp3, survives instance replacement) |
| repo | `/data/Mamba-ASR` |
| LibriSpeech | `/data/LibriSpeech` |
| work dir (CSVs, tokenizer, teacher features, runs) | `/data/work` |

## New files

| file | role |
|---|---|
| `train_distill_CTC.py` | training script (copy of `train_CTC.py` + teacher features, distillation loss, greedy test decoding, logging) |
| `hparams/CTC/conmamba_causal_distill.yaml` | causal student, 100 h, `distill_proj`, `kd_weight` |
| `prepare_distill_data.py` | CSVs, the 30 s filter, the 10 h and 100-utterance subsets, the shared tokenizer |
| `text_norm.py` | shared `normalize()` |
| `distill_utils.py` | frame-count math (student vs teacher) |
| `teacher/make_pseudo_labels.py` | teacher pass 1 (Whisper large-v3, faster-whisper) |
| `teacher/extract_features.py` | teacher pass 2 (Whisper-small encoder → 25 fps fp16 memmap + stats) |
| `teacher/plot_features.py` | sanity check 8 |
| `teacher/eval_whisper.py` | offline Whisper reference WER |
| `streaming.py` | step-by-step inference (GPU and CPU), equivalence check, speed benchmarks, emission frames |
| `scripts/setup_envs.sh` | creates both conda envs |
| `scripts/check_install.py` | installation check 2 (Mamba kernel timing) |
| `scripts/run_exp.sh` | launches E0 / E1 / E2 / E3 / E0-sub in the background |
| `scripts/collect_results.py` | WER and speed tables, curves |
| `tests/` | sanity checks 1–5 and 11 (pytest) |

### Deviations from the plan (already applied; log them in the report)

1. **CNN `padding: causal`.** In SpeechBrain 1.0.0 this setting pads the frequency
   axis, not time. The time axis gets no padding ("valid"), so student frame *i*
   uses mel frames 4i … 4i+6. That is streamable. The frame math in
   `distill_utils.py`, check 4 and `streaming.py` all use this exact receptive field.
2. **Check 4 is stricter than the plan.** Frames before `ceil((k-6)/4)` must stay
   unchanged, and frame `ceil((k-6)/4)` must change.
3. **The normalizer expands abbreviations and symbols** (`Mr.`→`MISTER`, `Mrs.`→`MISSUS`,
   `$5`→`FIVE DOLLARS`, `&`→`AND`). This runs before the plan's steps.
4. **Shared data folder.** CSVs live in `/data/work/data` and are built once by
   `prepare_distill_data.py`. The training runs use them read-only (`skip_prep` is
   unnecessary because the training script never calls the LibriSpeech prep).
5. **dev-clean is also in `test_csv`,** so the averaged model is reported on all four sets.
6. **Averaging only uses end-of-epoch checkpoints.** Intra-epoch checkpoints are
   excluded explicitly. The averaged and best models are exported to
   `runs/<name>/final/` for `streaming.py`.

---

## 0. Launch the instance (AWS console)

1. EC2 → Launch instance:
   - **AMI:** "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)". It comes
     with the NVIDIA driver; CUDA 11.8 build tools are installed by the setup script.
   - **Instance type:** `g5.xlarge`. Purchase option: **Spot**, persistent request,
     interruption behavior **Stop**.
   - **Storage:** root 100 GB, plus **a second gp3 volume of 200 GB** with
     "Delete on termination" **unchecked**.
2. Billing → Budgets: create alarms at **$60** and **$80**.
3. (Optional) Create an S3 bucket for backups, e.g. `s3://<you>-mamba-asr`, and attach
   an IAM role with S3 access to the instance.

## 1. One-time machine setup (SSH into the instance)

Mount the data volume. **Run `mkfs` only the very first time**; it erases the volume.

```bash
lsblk                                   # find the 200 GB device, e.g. /dev/nvme1n1
sudo mkfs.ext4 /dev/nvme1n1             # FIRST TIME ONLY
sudo mkdir -p /data && sudo mount /dev/nvme1n1 /data && sudo chown $USER /data
echo "/dev/nvme1n1 /data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

Install conda (Miniforge) and get the code onto the data volume:

```bash
cd /data
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /data/miniforge3 && /data/miniforge3/bin/conda init bash && exec bash
git clone <your fork URL> /data/Mamba-ASR
sudo apt-get install -y tmux
```

## 2. Create the two Python environments (installation checks 1–4)

```bash
cd /data/Mamba-ASR
bash scripts/setup_envs.sh train      # mamba-train: py3.9, torch 2.0.1 cu118, speechbrain 1.0.0, mamba-ssm 1.1.3.post1
bash scripts/setup_envs.sh teacher    # whisper-teacher: py3.10, recent torch, transformers, faster-whisper
```

The `train` step finishes by running `scripts/check_install.py`. It must print
`INSTALL CHECK: PASS`. One causal ConMamba layer should take a few ms for
(8, 1000, 256) forward and backward. If `mamba-ssm` or `causal-conv1d` fails to
build, adjust as little as you can, then record the working versions.

Record the environment in the run log:

```bash
nvidia-smi > /data/work_env.txt
conda run -n mamba-train pip freeze > /data/env_train.txt
conda run -n whisper-teacher pip freeze > /data/env_teacher.txt
```

Installation check 3 (teacher env loads both Whisper models) happens implicitly in
step 5's `--limit` runs.

## 3. Download LibriSpeech (about 7.5 GB)

```bash
cd /data
for s in train-clean-100 dev-clean dev-other test-clean test-other; do
  wget -c https://www.openslr.org/resources/12/$s.tar.gz && tar xzf $s.tar.gz && rm $s.tar.gz
done
ls /data/LibriSpeech     # train-clean-100 dev-clean dev-other test-clean test-other ...
```

## 4. Prepare data + shared tokenizer (training env)

```bash
conda activate mamba-train && cd /data/Mamba-ASR
python prepare_distill_data.py --data_folder /data/LibriSpeech --work_dir /data/work
```

This prints hours and utterance counts per CSV (also saved to
`/data/work/data/dataset_stats.json`), the number of rows removed by the 30 s
filter, and the 31 tokenizer entries. It fails if `normalize()` changes any
ground-truth row.

**Sanity checks 1 and 2:**

```bash
pytest -q tests/test_1_text_norm.py
WORK_DIR=/data/work pytest -q tests/test_2_tokenizer.py
```

## 5. Teacher passes (teacher env, run inside tmux)

```bash
tmux new -s teacher
conda activate whisper-teacher && cd /data/Mamba-ASR
```

**Pass 1: pseudo-labels.** Time 500 utterances first; it prints s/utt and an
extrapolated GPU-hour figure.

```bash
python teacher/make_pseudo_labels.py --train_csv /data/work/data/train_gt.csv --out_dir /data/work/data --limit 500
python teacher/make_pseudo_labels.py --train_csv /data/work/data/train_gt.csv --out_dir /data/work/data
```

Outputs: `train_pl.csv` (kept rows, for E1 and E3), `train_gt_plkept.csv` (for
E0-sub), `train_pl_meta.csv`, `pl_report.json` and `pl_dropped_sample.txt`
(20 examples to inspect). It resumes if interrupted. If
`pl_report.json` says `"run_E0_sub": true`, more than 5 % of hours were dropped,
so also run E0-sub.

**Pass 2: encoder features.**

```bash
# timing + check 8 on 500 utterances
python teacher/extract_features.py --train_csv /data/work/data/train_gt.csv --out_dir /data/work/teacher --limit 500
python teacher/plot_features.py --train_csv /data/work/data/train_gt.csv --teacher_dir /data/work/teacher \
    --suffix _limit500 --out /data/work/teacher/check8.png
# full extraction (~14 GB, resumable), then stats are computed automatically
python teacher/extract_features.py --train_csv /data/work/data/train_gt.csv --out_dir /data/work/teacher
# features for the 100-utterance overfitting set (check 7)
python teacher/extract_features.py --train_csv /data/work/data/train_gt_overfit100.csv --out_dir /data/work/teacher_overfit
```

**Check 8:** open `check8.png` (`scp` it to your laptop). The red feature-norm curve
should rise in speech and fall in silence, lined up with the waveform.

**Reference WERs** (can run now or at the end):

```bash
python teacher/eval_whisper.py --data_dir /data/work/data --model small    --out_dir /data/work/whisper_ref
python teacher/eval_whisper.py --data_dir /data/work/data --model large-v3 --out_dir /data/work/whisper_ref
```

## 6. Sanity checks 3–7 (training env)

```bash
conda activate mamba-train && cd /data/Mamba-ASR
WORK_DIR=/data/work pytest -q tests/test_3_frame_alignment.py   # check 3
pytest -q tests/test_4_causality.py                             # check 4
pytest -q tests/test_5_streaming.py                             # check 5 (random weights; trained later)
```

Common overrides for the overfitting tests (shell variable, used below):

```bash
W=/data/work; D=/data/LibriSpeech; O=$W/data/train_gt_overfit100.csv
OVERFIT="--work_dir $W --data_folder $D --train_csv $O --valid_csv $O --test_csv [$O] \
  --allow_train_as_valid True --use_fea_augment False --transformer_dropout 0.0 --cnn_dropout 0.0 \
  --constant_lr True --grad_accumulation_factor 1 --max_batch_length_train 100 \
  --number_of_epochs 200 --eval_modes [best] --max_projected_hours null"
```

**Check 6** (CTC only; about 2,000 steps):

```bash
python train_distill_CTC.py hparams/CTC/conmamba_causal_distill.yaml $OVERFIT \
  --experiment check6_overfit_ctc --kd_weight 0.0
```

**Check 7** (with distillation):

```bash
python train_distill_CTC.py hparams/CTC/conmamba_causal_distill.yaml $OVERFIT \
  --experiment check7_overfit_kd --kd_weight 1.0 --teacher_dir $W/teacher_overfit
```

Pass criteria: training-set WER < 5 %. Look at the last `valid WER` lines in
`$W/runs/check6_overfit_ctc/train_log.txt`, or `results_greedy.json`. For check 7,
the `cos` field in `step_log.jsonl` should rise steadily, ideally above 0.7.

## 7. Sanity checks 9, 10, 11 (first paid-time runs, about 1 GPU hour)

Checks 9 and 10 share one run: E0 on the 10 h subset for 5 epochs. The run is
killed at step 150 and then resumed.

```bash
MINI="--work_dir $W --data_folder $D --train_csv $W/data/train_gt_10h.csv --kd_weight 0.0 \
  --number_of_epochs 5 --experiment check9_10_mini --max_projected_hours null"
# run 1: intra-epoch checkpoint every 50 steps, simulated spot kill at step 150
python train_distill_CTC.py hparams/CTC/conmamba_causal_distill.yaml $MINI --ckpt_interval_steps 50 --kill_at_step 150
# run 2: same command without the kill -> must resume from the last checkpoint
python train_distill_CTC.py hparams/CTC/conmamba_causal_distill.yaml $MINI --ckpt_interval_steps 50
```

- **Check 9:** in `$W/runs/check9_10_mini/step_log.jsonl`, `step`, `lr` and `loss_ctc`
  continue smoothly across the restart.
- **Check 10:** dev-clean WER in `train_log.txt` falls every epoch.
- **Check 11:**

```bash
RUN_DIR=$W/runs/check9_10_mini pytest -q tests/test_11_checkpoint_avg.py
RUN_DIR=$W/runs/check9_10_mini pytest -q tests/test_5_streaming.py   # check 5 on a trained model
```

→ **Check-in with the project owner** (all checks passed) before full runs.

## 8. Full experiments (E0 → E2 → E1 → E3)

Run them one at a time, inside tmux. Use a fixed `RUN_NAME` per experiment and
reuse it unchanged to resume after an interruption.

```bash
conda activate mamba-train && cd /data/Mamba-ASR
export WORK_DIR=/data/work DATA_FOLDER=/data/LibriSpeech
export S3_URI=s3://<you>-mamba-asr/runs        # optional backup after every epoch
RUN_NAME=E0_20261106 bash scripts/run_exp.sh E0
tail -f /data/work/logs/E0_20261106.log
```

The script launches this command (shown so you can run it by hand):

```bash
python train_distill_CTC.py hparams/CTC/conmamba_causal_distill.yaml \
  --work_dir /data/work --data_folder /data/LibriSpeech --experiment E0_20261106 \
  --train_csv /data/work/data/train_gt.csv --kd_weight 0.0 --ckpt_interval_minutes 10
```

| ID | command |
|---|---|
| E0 | `RUN_NAME=E0_<date> bash scripts/run_exp.sh E0` (ground truth, no distillation) |
| E2 | `RUN_NAME=E2_<date> bash scripts/run_exp.sh E2` (ground truth, `kd_weight` 1.0) |
| E1 | `RUN_NAME=E1_<date> bash scripts/run_exp.sh E1` (pseudo-labels, no distillation) |
| E3 | `RUN_NAME=E3_<date> bash scripts/run_exp.sh E3` (pseudo-labels, `kd_weight` 1.0) |
| E0-sub | `RUN_NAME=E0sub_<date> bash scripts/run_exp.sh E0-sub` (only if `run_E0_sub` is true) |

**What to watch:**

- **At step 200:** the log prints a `[throughput]` line (also saved in
  `runs/<name>/throughput.json`) with s/step, projected GPU hours and peak memory.
  - If the projection exceeds 12 h, the run **stops itself**. Report the number;
    after approval, rerun with `--max_projected_hours null`.
  - If peak memory is above 22 GB, the log warns. Rerun with
    `--max_batch_length_train 250 --grad_accumulation_factor 3`.
- **Balance check (E2):** compare `loss_ctc` in `step_log.jsonl` against E0 over the
  first 2 epochs. If E2's CTC loss falls clearly slower, rerun E2 with `--kd_weight 0.3`
  (new `RUN_NAME`).
- **Every epoch:** `train_log.txt` shows dev-clean WER and CER, plus
  `unalignable_pct`. Above 0.1 % triggers a warning that points to a frame-rate
  or tokenizer bug. `fixed_dev_examples.txt` holds the same 5 dev-clean
  references and hypotheses each epoch.
- **Spot interruption:** start the instance again (or attach the volume to a new
  one and mount it), then rerun **the same** `RUN_NAME=... bash scripts/run_exp.sh ...`.

When training ends, the script automatically:
- evaluates the averaged model (up to 10 lowest-dev-clean-WER checkpoints) and the
  best checkpoint on dev-clean, dev-other, test-clean and test-other, using greedy
  CTC decoding;
- writes `runs/<name>/results_greedy.json` and `wer_<mode>_greedy_<set>.txt`;
- exports `runs/<name>/final/{model_avg,model_best,normalizer}.ckpt` and `info.json`.

To add an optional beam-search column (labelled as such), run this after training:
```bash
python train_distill_CTC.py hparams/CTC/conmamba_causal_distill.yaml --work_dir $W --data_folder $D \
  --experiment E0_20261106 --skip_train True --test_decoder beam
```

→ **Check-in after E0 and E2**, before E1 and E3.

## 9. Streaming, speed, emission frames (per final model)

```bash
R=/data/work/runs/E0_20261106
python streaming.py verify    --run_dir $R --n 5        # check 5 on the trained model: max|diff|<1e-3, same greedy text
python streaming.py bench     --run_dir $R              # -> $R/final/speed_avg.json
python streaming.py emissions --run_dir $R --split dev-clean   # -> token frame indices for later delay work
```

`bench` reports parameter count and fp32 size (excluding `distill_proj`), plus:
- GPU offline real-time factor (full sequence, batch 1, 200 dev-clean utterances);
- GPU streaming milliseconds per 40 ms frame;
- single-thread CPU streaming milliseconds per 40 ms frame.

## 10. Report

```bash
python scripts/collect_results.py --work_dir /data/work --runs E0_20261106 E2_20261106 E1_20261106 E3_20261106
```

This writes `/data/work/report/results.md` (WER table including the Whisper
references, the speed table and pseudo-label statistics) and `curves.png`
(CTC loss, distillation loss, cosine and dev-clean WER).

## Cost hygiene

- `sudo shutdown -h now` whenever no job is running. Never leave the GPU idle overnight.
- Always run inside `tmux` or through `scripts/run_exp.sh` (`nohup`), with logs in `/data/work/logs`.
- Report spend at $20, $40 and $60; stop at $80.
- Do not run the teacher and a training job on the GPU at the same time. Training
  can use most of the 24 GB.
