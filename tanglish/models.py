"""Classifier heads over a pretrained encoder.

head="cls"   sentence only: the [CLS] vector.
head="word"  word only: subword vectors are mean-pooled into words, then
             additive attention over words gives one vector.
head="hier"  hierarchical fusion: subword -> word -> sentence, with a learned
             gate mixing the [CLS] vector and the word-attention vector.

label_tree=True replaces the flat 6-way output with OLID-style heads that
follow the label hierarchy (see LABEL_TREE) and composes them into 6-class
log-probabilities. Every model returns `scores` such that softmax(scores) is
the class distribution, so evaluation and ensembling treat them alike.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

NUM_CLASSES = 6

# Level A: 0 not offensive, 1 offensive, 2 not Tamil.
# Level B (offensive only): 0 untargeted, 1 targeted.
# Level C (targeted only): 0 individual, 1 group, 2 other.
# For every 6-way label id: (A, B or -1, C or -1).
LABEL_TREE = {
    0: (0, -1, -1),
    1: (1, 0, -1),
    2: (1, 1, 0),
    3: (1, 1, 1),
    4: (1, 1, 2),
    5: (2, -1, -1),
}


def tree_targets(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    table = torch.tensor([LABEL_TREE[i] for i in range(NUM_CLASSES)], device=labels.device)
    rows = table[labels]
    return rows[:, 0], rows[:, 1], rows[:, 2]


def compose_tree(logits_a, logits_b, logits_c) -> torch.Tensor:
    """log P(y) = log P(A) + log P(B|A) + log P(C|B) along each label's path."""
    la, lb, lc = (F.log_softmax(x.float(), dim=-1) for x in (logits_a, logits_b, logits_c))
    targeted = la[:, 1] + lb[:, 1]
    return torch.stack(
        [la[:, 0], la[:, 1] + lb[:, 0], targeted + lc[:, 0], targeted + lc[:, 1], targeted + lc[:, 2], la[:, 2]],
        dim=-1,
    )


def pool_words(hidden: torch.Tensor, word_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pools subword vectors into word vectors.

    hidden: (B, L, H). word_index: (B, L) word id per token, -1 for special
    tokens and padding. Returns word vectors (B, W, H) and a mask (B, W).
    """
    b, _, h = hidden.shape
    n_words = int(word_index.max().item()) + 1 if word_index.numel() else 0
    n_words = max(n_words, 1)
    valid = (word_index >= 0).unsqueeze(-1).to(hidden.dtype)
    idx = word_index.clamp(min=0)
    sums = hidden.new_zeros(b, n_words, h).scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, h), hidden * valid)
    counts = hidden.new_zeros(b, n_words).scatter_add_(1, idx, valid.squeeze(-1))
    return sums / counts.clamp(min=1).unsqueeze(-1), counts > 0


class WordAttention(nn.Module):
    """Additive attention over word vectors (Yang et al., 2016)."""

    def __init__(self, hidden: int, attn_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden, attn_dim)
        self.context = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, words: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.context(torch.tanh(self.proj(words))).squeeze(-1)
        scores = scores.float().masked_fill(~mask, torch.finfo(torch.float32).min)
        weights = torch.softmax(scores, dim=-1)
        # A row with no words (can only happen on empty input) gets all-zero weights.
        weights = weights * mask
        pooled = torch.bmm(weights.unsqueeze(1).to(words.dtype), words).squeeze(1)
        return pooled, weights


def load_encoder(model_name: str) -> nn.Module:
    # The pooler is unused; dropping it avoids unused-parameter errors under DDP.
    try:
        return AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    except TypeError:
        return AutoModel.from_pretrained(model_name)


class OffensiveClassifier(nn.Module):
    def __init__(self, model_name: str, head: str = "hier", label_tree: bool = False,
                 dropout: float = 0.1, attn_dim: int = 256):
        super().__init__()
        if head not in ("cls", "word", "hier"):
            raise ValueError(f"unknown head {head!r}")
        self.head = head
        self.label_tree = label_tree
        self.encoder = load_encoder(model_name)
        hidden = self.encoder.config.hidden_size

        if head in ("word", "hier"):
            self.word_attn = WordAttention(hidden, attn_dim)
        if head == "hier":
            self.gate = nn.Linear(2 * hidden, hidden)
        self.dropout = nn.Dropout(dropout)

        if label_tree:
            self.out_a = nn.Linear(hidden, 3)
            self.out_b = nn.Linear(hidden, 2)
            self.out_c = nn.Linear(hidden, 3)
        else:
            self.out = nn.Linear(hidden, NUM_CLASSES)

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def represent(self, input_ids, attention_mask, word_index=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_vec = hidden[:, 0]
        attn = None
        if self.head == "cls":
            return cls_vec, attn
        words, word_mask = pool_words(hidden, word_index)
        word_vec, attn = self.word_attn(words, word_mask)
        if self.head == "word":
            return word_vec, attn
        gate = torch.sigmoid(self.gate(torch.cat([cls_vec, word_vec], dim=-1)))
        return gate * cls_vec + (1 - gate) * word_vec, attn

    def forward(self, input_ids, attention_mask, word_index=None):
        """Returns a dict with `scores` (B, 6), level logits if label_tree, and
        `attention` (B, W) word weights for the word/hier heads."""
        rep, attn = self.represent(input_ids, attention_mask, word_index)
        rep = self.dropout(rep)
        out = {"attention": attn}
        if self.label_tree:
            out["logits_a"], out["logits_b"], out["logits_c"] = self.out_a(rep), self.out_b(rep), self.out_c(rep)
            out["scores"] = compose_tree(out["logits_a"], out["logits_b"], out["logits_c"])
        else:
            out["scores"] = self.out(rep).float()
        return out
