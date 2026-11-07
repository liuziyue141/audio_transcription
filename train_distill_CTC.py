#!/usr/bin/env python3
"""Whisper-teacher -> causal ConMamba CTC student (baseline E0-E3).

Copy of train_CTC.py with:
  * shared, pre-trained character tokenizer (never re-trained per run)
  * cached Whisper encoder features as a training-only dynamic item
  * L1 + cosine (DistilHuBERT) frame-level distillation through distill_proj
  * CTC and distillation computed in fp32
  * greedy CTC decoding in TEST (beam search only if test_decoder: beam)
  * per-50-step logging, unalignable-utterance counting, throughput check
  * checkpoint averaging that excludes intra-epoch checkpoints, and export
    of the averaged model for streaming.py

Usage:
  python train_distill_CTC.py hparams/CTC/conmamba_causal_distill.yaml \
      --work_dir /data/work --data_folder /data/LibriSpeech \
      --experiment E0_20261106 --ckpt_interval_minutes 10
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import if_main_process, run_on_main

try:
    from speechbrain.core import INTRA_EPOCH_CKPT_FLAG
except ImportError:  # pragma: no cover
    INTRA_EPOCH_CKPT_FLAG = "brain_intra_epoch_ckpt"

logger = logging.getLogger(__name__)

EVAL_CSV_NAMES = {"dev-clean.csv", "dev-other.csv", "test-clean.csv", "test-other.csv"}


# --------------------------------------------------------------------------
# Teacher features
# --------------------------------------------------------------------------
class TeacherFeatureStore:
    """Reads one utterance's standardized teacher features from the memmap."""

    def __init__(self, bin_path, index_path, stats_path, dim):
        with open(index_path) as f:
            idx = json.load(f)
        assert idx["dim"] == dim, f"teacher dim {idx['dim']} != {dim}"
        self.utts = idx["utts"]
        self.total = idx["total_frames"]
        self.dim = dim
        self.bin_path = bin_path
        st = np.load(stats_path)
        self.mean = st["mean"].astype(np.float32)
        self.std = np.maximum(st["std"].astype(np.float32), 1e-5)
        self._mm = None  # opened lazily (once per dataloader worker)

    def __call__(self, utt_id):
        if self._mm is None:
            self._mm = np.memmap(self.bin_path, dtype=np.float16, mode="r",
                                 shape=(self.total, self.dim))
        off, n = self.utts[utt_id]
        x = np.asarray(self._mm[off: off + n], dtype=np.float32)
        return torch.from_numpy((x - self.mean) / self.std)


def distill_loss(s, s_rel, t, t_rel, max_diff):
    """L1 + -log sigmoid(cos) per valid frame (DistilHuBERT), fp32.

    s: (B, Ts, D) student projection; t: (B, Tt, D) standardized teacher.
    *_rel: SpeechBrain relative lengths. Returns (loss, mean_cos, max_len_diff).
    """
    s_len = torch.round(s_rel * s.shape[1]).long()
    t_len = torch.round(t_rel * t.shape[1]).long()
    diff = int((s_len - t_len).abs().max().item())
    if diff > max_diff:
        raise RuntimeError(
            f"student/teacher frame counts differ by {diff} > {max_diff}: "
            f"student={s_len.tolist()} teacher={t_len.tolist()} (frame-math bug)")
    L = torch.minimum(s_len, t_len)
    T = int(L.max().item())
    s = s[:, :T].float()
    t = t[:, :T].float()
    mask = (torch.arange(T, device=s.device)[None, :] < L[:, None]).float()
    l1 = (s - t).abs().mean(-1)
    cos = F.cosine_similarity(s, t, dim=-1)
    per_frame = l1 - F.logsigmoid(cos)
    n = mask.sum().clamp(min=1.0)
    return (per_frame * mask).sum() / n, ((cos * mask).sum() / n).detach(), diff


# --------------------------------------------------------------------------
# Brain
# --------------------------------------------------------------------------
class ASR(sb.core.Brain):
    # ---- forward ---------------------------------------------------------
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        feats = self.hparams.compute_features(wavs)  # (B, T, 80)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        if stage == sb.Stage.TRAIN and self.hparams.use_fea_augment:
            n_before = feats.shape[0]
            feats, _ = self.hparams.fea_augment(feats, wav_lens)
            assert feats.shape[0] == n_before, "augmenter changed the batch size"

        src = self.modules.CNN(feats)
        enc_out, _ = self.modules.Transformer(src, tgt=None, wav_len=wav_lens)

        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        s_proj = None
        if stage == sb.Stage.TRAIN and self.kd_on:
            s_proj = self.modules.distill_proj(enc_out)

        p_tokens = None
        if stage != sb.Stage.TRAIN:
            if stage == sb.Stage.TEST and self.hparams.test_decoder == "beam":
                p_tokens = self.test_searcher(p_ctc, wav_lens)
            else:
                p_tokens = sb.decoders.ctc_greedy_decode(
                    p_ctc, wav_lens, blank_id=self.hparams.blank_index)
        return p_ctc, wav_lens, p_tokens, s_proj

    # ---- losses ----------------------------------------------------------
    def compute_objectives(self, predictions, batch, stage):
        p_ctc, wav_lens, predicted_tokens, s_proj = predictions
        ids = batch.id
        tokens, tokens_lens = batch.tokens

        if stage == sb.Stage.TRAIN and self.hparams.use_fea_augment:
            tokens = self.hparams.fea_augment.replicate_labels(tokens)
            tokens_lens = self.hparams.fea_augment.replicate_labels(tokens_lens)

        dev_type = torch.device(self.device).type
        with torch.autocast(device_type=dev_type, enabled=False):
            loss_ctc = self.hparams.ctc_cost(
                p_ctc.float(), tokens, wav_lens, tokens_lens).sum()
            loss = loss_ctc
            if stage == sb.Stage.TRAIN:
                self._count_unalignable(p_ctc.shape[1], wav_lens, tokens, tokens_lens)
                stats = {"loss_ctc": loss_ctc.detach()}
                if s_proj is not None:
                    t, t_rel = batch.teacher_feats
                    l_kd, cos, diff = distill_loss(
                        s_proj, wav_lens, t, t_rel, self.hparams.max_frame_diff)
                    loss = loss + self.hparams.kd_weight * l_kd
                    stats["loss_distill"] = l_kd.detach()
                    stats["cos"] = cos
                    stats["max_frame_diff"] = torch.tensor(float(diff))
                stats["loss"] = loss.detach()
                for k, v in stats.items():
                    self._acc[k] = self._acc.get(k, 0.0) + v.float()
                self._acc_n += 1

        if stage != sb.Stage.TRAIN:
            if stage == sb.Stage.TEST and self.hparams.test_decoder == "beam":
                predicted_words = [hyp[0].text.split(" ") for hyp in predicted_tokens]
            else:
                predicted_words = self.tokenizer(predicted_tokens, task="decode_from_list")
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
            if stage == sb.Stage.VALID:
                for uid, hyp, ref in zip(ids, predicted_words, target_words):
                    if uid in self.fixed_ids:
                        self._fixed_out[uid] = (" ".join(ref), " ".join(hyp))
        return loss

    def _count_unalignable(self, T, wav_lens, tokens, tokens_lens):
        """Utterances whose chars + repeated pairs exceed their output frames."""
        frames = torch.round(wav_lens * T).long()
        tl = torch.round(tokens_lens * tokens.shape[1]).long()
        valid = torch.arange(tokens.shape[1], device=tokens.device)[None] < tl[:, None]
        rep = ((tokens[:, 1:] == tokens[:, :-1]) & valid[:, 1:]).sum(1)
        need = tl + rep
        self._unalign += int((need > frames).sum().item())
        self._epoch_utts += tokens.shape[0]

    # ---- grad norm (logged, not changed) ---------------------------------
    def optimizers_step(self, *args, **kwargs):
        grads = [p.grad.detach().float().norm() for p in self.modules.parameters()
                 if p.grad is not None]
        self._last_grad_norm = (torch.norm(torch.stack(grads)).item() if grads else 0.0)
        return super().optimizers_step(*args, **kwargs)

    # ---- hooks -----------------------------------------------------------
    def on_fit_start(self):
        super().on_fit_start()
        self._acc, self._acc_n = {}, 0
        self._last_grad_norm = None
        self._t_mark = None

    def on_stage_start(self, stage, epoch):
        if stage == sb.Stage.TRAIN:
            self._unalign, self._epoch_utts = 0, 0
            self._acc, self._acc_n = {}, 0
        else:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.wer_computer()
            self._fixed_out = {}

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        if not should_step:
            return
        if not self.hparams.constant_lr:
            self.hparams.noam_annealing(self.optimizer)
        step = self.optimizer_step
        now = time.time()
        if step == 20:
            self._t_mark = now
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        if step % self.hparams.log_every_steps == 0:
            self._log_step(step)
        if step == self.hparams.throughput_check_step and self._t_mark is not None:
            self._throughput_check(step, now)
        if self.hparams.kill_at_step and step == self.hparams.kill_at_step:
            logger.warning(f"kill_at_step={step}: simulating a spot interruption")
            logging.shutdown()
            os._exit(17)

    def _log_step(self, step):
        if self._acc_n == 0:
            return
        rec = {k: round(float(v) / self._acc_n, 6) for k, v in self._acc.items()}
        rec.update({
            "step": step,
            "epoch": self.hparams.epoch_counter.current,
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm_pre_clip": self._last_grad_norm,
            "time": time.time(),
        })
        self._acc, self._acc_n = {}, 0
        if if_main_process():
            with open(self.hparams.step_log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if self.hparams.use_wandb:
                import wandb
                if wandb.run is not None:
                    wandb.log({f"step/{k}": v for k, v in rec.items() if v is not None}, step=step)

    def _throughput_check(self, step, now):
        sec_per_step = (now - self._t_mark) / (step - 20)
        total_steps = self.steps_per_epoch * self.hparams.number_of_epochs
        proj_h = sec_per_step * total_steps / 3600.0
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        msg = (f"[throughput] {sec_per_step:.3f} s/step, {self.steps_per_epoch} steps/epoch, "
               f"projected {proj_h:.2f} GPU h for {self.hparams.number_of_epochs} epochs "
               f"(train only, excl. validation); peak GPU mem {peak:.1f} GB")
        logger.info(msg)
        print(msg, flush=True)
        if if_main_process():
            with open(os.path.join(self.hparams.output_folder, "throughput.json"), "w") as f:
                json.dump({"sec_per_step": sec_per_step, "steps_per_epoch": self.steps_per_epoch,
                           "projected_train_hours": proj_h, "peak_mem_gb": peak}, f, indent=2)
        if peak > 22.0:
            logger.warning("peak GPU memory > 22 GB: lower max_batch_length_train to 250 and "
                           "raise grad_accumulation_factor to keep ~600 s per update")
        limit = self.hparams.max_projected_hours
        if limit is not None and proj_h > limit:
            print(f"[throughput] projected {proj_h:.1f} h > max_projected_hours={limit}: "
                  "stopping. Report to the project owner; rerun with "
                  "--max_projected_hours null once approved.", flush=True)
            logging.shutdown()
            os._exit(3)

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            pct = 100.0 * self._unalign / max(1, self._epoch_utts)
            stage_stats["unalignable"] = self._unalign
            stage_stats["unalignable_pct"] = round(pct, 4)
            if pct > 0.1:
                logger.warning(f"{pct:.3f}% of utterances cannot be CTC-aligned (> 0.1%): "
                               "frame-rate or tokenizer problem")
            self.train_stats = stage_stats
            return

        stage_stats["CER"] = self.cer_metric.summarize("error_rate")
        stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            lr = self.optimizer.param_groups[0]["lr"]
            epoch_stats = {"epoch": epoch, "lr": lr, "steps": self.optimizer_step,
                           "optimizer": self.optimizer.__class__.__name__}
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats, train_stats=self.train_stats, valid_stats=stage_stats)
            if if_main_process():
                with open(os.path.join(self.hparams.output_folder, "fixed_dev_examples.txt"), "a") as f:
                    f.write(f"=== epoch {epoch}  dev-clean WER {stage_stats['WER']:.2f}\n")
                    for uid in sorted(self._fixed_out):
                        ref, hyp = self._fixed_out[uid]
                        f.write(f"{uid}\n  REF: {ref}\n  HYP: {hyp}\n")
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"], "epoch": epoch},
                min_keys=["WER"],
                num_to_keep=self.hparams.avg_checkpoints,
            )
            self._s3_sync()

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current,
                            "mode": self.eval_mode, "set": self.test_name},
                test_stats=stage_stats)
            self.test_results.setdefault(self.eval_mode, {})[self.test_name] = {
                "WER": stage_stats["WER"], "CER": stage_stats["CER"]}
            if if_main_process():
                with open(self.hparams.wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

    def _s3_sync(self):
        uri = self.hparams.s3_uri
        if not uri or not if_main_process():
            return
        try:
            subprocess.Popen(["aws", "s3", "sync", self.hparams.output_folder, uri,
                              "--only-show-errors"])
        except OSError as e:
            logger.warning(f"s3 sync failed to start: {e}")

    # ---- evaluation: averaging that never sees distill_proj / intra-epoch ckpts
    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        ckpts = self.checkpointer.find_checkpoints(
            min_key=min_key, max_key=max_key,
            ckpt_predicate=lambda c: INTRA_EPOCH_CKPT_FLAG not in c.meta and "WER" in c.meta,
            max_num_checkpoints=1 if self.eval_mode == "best" else None,
        )
        assert ckpts, "no end-of-epoch checkpoints with a WER found"
        avg = sb.utils.checkpoints.average_checkpoints(ckpts, recoverable_name="model")
        bad = [k for k in avg if "distill_proj" in k or "teacher" in k]
        assert not bad, f"averaged model contains training-only keys: {bad[:5]}"
        self.hparams.model.load_state_dict(avg, strict=True)
        self.hparams.model.eval()
        used = [{"epoch": c.meta.get("epoch"), "WER": c.meta.get("WER"),
                 "path": str(c.path)} for c in ckpts]
        logger.info(f"[{self.eval_mode}] loaded model from {len(ckpts)} checkpoint(s): {used}")
        if self.eval_mode == "average" and not self._exported and if_main_process():
            out = Path(self.hparams.output_folder) / "final"
            out.mkdir(parents=True, exist_ok=True)
            torch.save(self.hparams.model.state_dict(), out / "model_avg.ckpt")
            # ckpts is sorted best-first (lowest dev-clean WER)
            torch.save(sb.utils.checkpoints.average_checkpoints(ckpts[:1], recoverable_name="model"),
                       out / "model_best.ckpt")
            shutil.copy(ckpts[0].paramfiles["normalizer"], out / "normalizer.ckpt")
            with open(out / "info.json", "w") as f:
                json.dump({"averaged_checkpoints": used,
                           "selection": "lowest dev-clean WER (min_key=WER), end-of-epoch only"},
                          f, indent=2)
            self._exported = True


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def dataio_prepare(hparams, tokenizer, teacher_store):
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(csv_path=hparams["train_csv"])
    if hparams["sorting"] == "ascending":
        train_data = train_data.filtered_sorted(sort_key="duration")
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] != "random":
        raise NotImplementedError("sorting must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(csv_path=hparams["valid_csv"])
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file).filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data] + list(test_datasets.values())
    valtest = [valid_data] + list(test_datasets.values())

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)  # no speed perturbation

    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides("wrd", "char_list", "tokens_list", "tokens")
    def text_pipeline(wrd):
        yield wrd
        yield list(wrd)
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        yield torch.LongTensor(tokens_list)

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    keys = ["id", "sig", "wrd", "char_list", "tokens"]
    sb.dataio.dataset.set_output_keys(valtest, keys)
    if teacher_store is not None:
        @sb.utils.data_pipeline.takes("id")
        @sb.utils.data_pipeline.provides("teacher_feats")
        def teacher_pipeline(uid):
            return teacher_store(uid)

        sb.dataio.dataset.add_dynamic_item([train_data], teacher_pipeline)
        sb.dataio.dataset.set_output_keys([train_data], keys + ["teacher_feats"])
    else:
        sb.dataio.dataset.set_output_keys([train_data], keys)

    train_bs = valid_bs = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler

        train_bs = DynamicBatchSampler(train_data, length_func=lambda x: x["duration"],
                                       **hparams["dynamic_batch_sampler_train"])
        valid_bs = DynamicBatchSampler(valid_data, length_func=lambda x: x["duration"],
                                       **hparams["dynamic_batch_sampler_val"])
    return train_data, valid_data, test_datasets, train_bs, valid_bs


def read_csv_column(path, col):
    import csv
    with open(path, newline="") as f:
        return [r[col] for r in csv.DictReader(f)]


def startup_assertions(hparams):
    """Fixed settings that must hold for every run."""
    assert hparams["speed_perturb"] is False, "speed_perturb must be False"
    aug = hparams["fea_augment"]
    assert getattr(aug, "concat_original", False) is False, "concat_original must be False"
    assert getattr(aug, "repeat_augment", 1) == 1, "repeat_augment must be 1"
    assert hparams["normalize"].norm_type == "global", "norm_type must be global"
    assert hparams["Transformer"].causal is True, "Transformer causal must be True"
    assert hparams["mamba_config"]["bidirectional"] is False

    # references only ever come from the dev/test CSVs
    if not hparams["allow_train_as_valid"]:
        for p in [hparams["valid_csv"]] + list(hparams["test_csv"]):
            assert os.path.basename(p) in EVAL_CSV_NAMES, f"{p} is not a LibriSpeech dev/test CSV"
            assert os.path.abspath(p) != os.path.abspath(hparams["train_csv"])


def write_run_info(hparams, extra):
    info = dict(extra)
    repo = os.path.dirname(os.path.abspath(__file__))
    for key, cmd in {
        "git_commit": ["git", "-C", repo, "rev-parse", "HEAD"],
        "git_dirty": ["git", "-C", repo, "status", "--porcelain"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "nvidia_smi": ["nvidia-smi"],
    }.items():
        try:
            info[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        except Exception as e:  # noqa: BLE001
            info[key] = f"unavailable: {e}"
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    info["speechbrain"] = sb.__version__
    info["argv"] = sys.argv
    with open(os.path.join(hparams["output_folder"], "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.utils.distributed.ddp_init_group(run_opts)
    startup_assertions(hparams)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # ---- shared tokenizer (must already exist: prepare_distill_data.py)
    model_file = os.path.join(hparams["tokenizer_folder"],
                              f"{hparams['output_neurons']}_{hparams['token_type']}.model")
    if not os.path.isfile(model_file):
        raise SystemExit(f"{model_file} missing: run prepare_distill_data.py first "
                         "(the tokenizer is trained once, never per experiment)")
    tokenizer = SentencePiece(
        model_dir=hparams["tokenizer_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["tokenizer_train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        bos_id=hparams["bos_index"],
        eos_id=hparams["eos_index"],
    )
    assert tokenizer.sp.vocab_size() == 31, f"tokenizer size {tokenizer.sp.vocab_size()} != 31"
    train_chars = set("".join(read_csv_column(hparams["train_csv"], "wrd"))) - {" "}
    missing = [c for c in train_chars if tokenizer.sp.piece_to_id(c) == tokenizer.sp.unk_id()]
    assert not missing, f"characters not covered by the tokenizer: {missing}"

    # ---- teacher features
    kd_on = float(hparams["kd_weight"]) > 0
    teacher_store = None
    if kd_on:
        teacher_store = TeacherFeatureStore(
            hparams["teacher_feats"], hparams["teacher_index"],
            hparams["teacher_stats"], hparams["teacher_dim"])
        ids = read_csv_column(hparams["train_csv"], "ID")
        absent = [i for i in ids if i not in teacher_store.utts]
        assert not absent, f"{len(absent)} train IDs have no teacher features, e.g. {absent[:3]}"

    train_data, valid_data, test_datasets, train_bs, valid_bs = dataio_prepare(
        hparams, tokenizer, teacher_store)

    if hparams["use_wandb"]:
        hparams["train_logger"] = hparams["wandb_logger"]()

    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["model_opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.tokenizer = tokenizer
    asr_brain.kd_on = kd_on
    asr_brain.eval_mode = "average"
    asr_brain.test_name = None
    asr_brain.test_results = {}
    asr_brain._exported = False
    asr_brain.fixed_ids = set(sorted(read_csv_column(hparams["valid_csv"], "ID"))[:5])
    n_batches = len(train_bs) if train_bs is not None else len(train_data) // hparams["batch_size"]
    # grad_accumulation_factor is a SpeechBrain run option: a CLI value overrides
    # the YAML one inside Brain, so read the Brain's effective value.
    asr_brain.steps_per_epoch = max(1, n_batches // asr_brain.grad_accumulation_factor)

    if hparams["test_decoder"] == "beam":
        from speechbrain.decoders.ctc import CTCBeamSearcher
        vocab_list = [tokenizer.sp.id_to_piece(i) for i in range(tokenizer.sp.vocab_size())]
        asr_brain.test_searcher = CTCBeamSearcher(**hparams["test_beam_search"], vocab_list=vocab_list)

    n_model = sum(p.numel() for p in hparams["model"].parameters())
    n_proj = sum(p.numel() for p in hparams["distill_proj"].parameters())
    print(f"[params] model (exported, excl. distill_proj): {n_model/1e6:.2f} M; "
          f"distill_proj: {n_proj/1e6:.3f} M; fp32 size {n_model*4/2**20:.1f} MB", flush=True)
    if if_main_process():
        write_run_info(hparams, {"params_model": n_model, "params_distill_proj": n_proj,
                                 "kd_weight": hparams["kd_weight"], "train_csv": hparams["train_csv"],
                                 "train_utts": len(train_data)})

    train_opts = hparams["train_dataloader_opts"]
    valid_opts = hparams["valid_dataloader_opts"]
    if train_bs is not None:
        train_opts = {"batch_sampler": train_bs, "num_workers": hparams["num_workers"]}
    if valid_bs is not None:
        valid_opts = {"batch_sampler": valid_bs}

    if not hparams["skip_train"]:
        asr_brain.fit(asr_brain.hparams.epoch_counter, train_data, valid_data,
                      train_loader_kwargs=train_opts, valid_loader_kwargs=valid_opts)

    for mode in hparams["eval_modes"]:
        asr_brain.eval_mode = mode
        for name, ds in test_datasets.items():
            asr_brain.test_name = name
            asr_brain.hparams.wer_file = os.path.join(
                hparams["output_folder"], f"wer_{mode}_{hparams['test_decoder']}_{name}.txt")
            asr_brain.evaluate(ds, min_key="WER", test_loader_kwargs=hparams["test_dataloader_opts"])

    if if_main_process():
        with open(os.path.join(hparams["output_folder"],
                               f"results_{hparams['test_decoder']}.json"), "w") as f:
            json.dump(asr_brain.test_results, f, indent=2)
        print(json.dumps(asr_brain.test_results, indent=2))
