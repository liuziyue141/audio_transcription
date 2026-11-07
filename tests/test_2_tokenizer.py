"""Sanity check 2: tokenizer round trip on 100 normalized transcripts."""

import csv
import os
import random

import pytest

from text_norm import normalize


def test_tokenizer_round_trip(work_dir):
    spm = pytest.importorskip("sentencepiece")
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(work_dir, "tokenizer", "31_char.model"))
    assert sp.vocab_size() == 31

    with open(os.path.join(work_dir, "data", "train_gt.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    sample = random.Random(0).sample(rows, min(100, len(rows)))
    texts = [normalize(r["wrd"]) for r in sample]

    pl = os.path.join(work_dir, "data", "train_pl.csv")
    if os.path.exists(pl):
        with open(pl, newline="") as f:
            pl_rows = list(csv.DictReader(f))
        texts += [r["wrd"] for r in random.Random(1).sample(pl_rows, min(100, len(pl_rows)))]
        chars = set("".join(r["wrd"] for r in pl_rows)) - {" "}
        missing = [c for c in chars if sp.piece_to_id(c) == sp.unk_id()]
        assert not missing, f"pseudo-label characters not in tokenizer: {missing}"

    for t in texts:
        ids = sp.encode_as_ids(t)
        assert sp.unk_id() not in ids, t
        assert sp.decode_ids(ids) == t
