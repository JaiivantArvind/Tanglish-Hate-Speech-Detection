"""CPU unit tests. Run: python -m pytest tests -q"""

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from tanglish.config import load_config
from tanglish.data import LABELS, load_splits, text_key
from tanglish.evaluate import log_softmax, macro_f1, tune_offsets
from tanglish.lexicon import Lexicon, load_curated
from tanglish.losses import LossComputer, class_weights
from tanglish.models import LABEL_TREE, compose_tree, pool_words, tree_targets
from tanglish.preprocess import normalize
from tanglish.translit import script_of, tamil_to_roman

ROOT = Path(__file__).resolve().parents[1]
HAS_DATA = (ROOT / "data" / "DravidianCodeMix" / "tamil_offensive_full_train.csv").exists()


@pytest.mark.skipif(not HAS_DATA, reason="dataset not downloaded (run scripts/phase1_data.py)")
def test_splits_have_six_classes_and_no_leakage():
    s = load_splits(ROOT / "data")
    assert [len(s[k]) for k in ("dev", "test")] == [4388, 4392]
    for split in s.values():
        assert set(split["label"]) == set(range(len(LABELS)))
    held_out = set(s["dev"]["text"].map(text_key)) | set(s["test"]["text"].map(text_key))
    train_keys = s["train"]["text"].map(text_key)
    assert not train_keys.isin(held_out).any()
    assert not train_keys.duplicated().any()


def test_pool_words_ignores_special_and_padding():
    hidden = torch.arange(6, dtype=torch.float).view(1, 6, 1).repeat(2, 1, 1)
    # [CLS] w0 w0 w1 [SEP] PAD  /  [CLS] w0 [SEP] PAD PAD PAD
    word_index = torch.tensor([[-1, 0, 0, 1, -1, -1], [-1, 0, -1, -1, -1, -1]])
    words, mask = pool_words(hidden, word_index)
    assert words.shape == (2, 2, 1)
    assert words[0, :, 0].tolist() == [1.5, 3.0]
    assert words[1, 0, 0].item() == 1.0
    assert mask.tolist() == [[True, True], [True, False]]


def test_label_tree_composes_to_a_distribution():
    torch.manual_seed(0)
    logp = compose_tree(torch.randn(4, 3), torch.randn(4, 2), torch.randn(4, 3))
    assert torch.allclose(logp.exp().sum(-1), torch.ones(4), atol=1e-5)


def test_tree_targets_follow_the_hierarchy():
    a, b, c = tree_targets(torch.arange(6))
    assert list(zip(a.tolist(), b.tolist(), c.tolist())) == [LABEL_TREE[i] for i in range(6)]


def test_tree_loss_is_finite_with_missing_levels():
    labels = torch.tensor([0, 5, 0])  # no offensive examples: levels B and C are empty
    loss_fn = LossComputer([0, 1, 2, 3, 4, 5], label_tree=True, kind="ce", weight_mode="sqrt")
    out = {"logits_a": torch.randn(3, 3, requires_grad=True), "logits_b": torch.randn(3, 2), "logits_c": torch.randn(3, 3)}
    loss = loss_fn(out, labels)
    assert torch.isfinite(loss)


def test_class_weights_modes():
    labels = [0] * 90 + [1] * 10
    bal, sq = class_weights(labels, 2, "balanced"), class_weights(labels, 2, "sqrt")
    assert bal[1] / bal[0] == pytest.approx(9.0)
    assert sq[1] / sq[0] == pytest.approx(3.0)
    assert class_weights(labels, 3, "sqrt")[2] == 0  # absent class


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 6, 500), rng.integers(0, 6, 500)
    assert macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro", labels=list(range(6)), zero_division=0))


def test_tune_offsets_never_hurts_dev():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 6, 400)
    scores = rng.normal(size=(400, 6)) + np.eye(6)[y] * 0.8
    offsets = tune_offsets(scores, y)
    assert macro_f1(y, (log_softmax(scores) + offsets).argmax(1)) >= macro_f1(y, scores.argmax(1))


def test_normalize():
    assert normalize("Semma 😂😂😂 @Rajini_fan #Thalaivar loooosu") == \
        "Semma face with tears of joy face with tears of joy @user Thalaivar loosu"
    assert normalize(None) == ""


def test_tamil_to_roman_and_script():
    assert tamil_to_roman("வெற்றி பெற வாழ்த்துக்கள்") == "vetri pera vaazhththukkal"
    assert tamil_to_roman("நன்றி") == "nandri"
    assert script_of("கோமாளி da") == "tamil" and script_of("padam semma") == "roman"


def test_curated_lexicon_tags_words():
    lex = Lexicon(load_curated())
    assert lex.apply("dei loosu poda") == "dei loosu <LEX_STUPID> poda"
    assert "dei" not in lex.terms and "panni" not in lex.terms  # ambiguous terms excluded


def test_config_keeps_off_as_string(tmp_path):
    (tmp_path / "base.yaml").write_text("lexicon: off\nepochs: 3\n")
    (tmp_path / "exp.yaml").write_text("inherit: base.yaml\nname: x\n")
    cfg = load_config(str(tmp_path / "exp.yaml"), ["translit=off", "lr=1e-5"])
    assert (cfg.lexicon, cfg.translit, cfg.epochs, cfg.name, cfg.lr) == ("off", "off", 3, "x", 1e-5)
