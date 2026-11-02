"""Experiment configuration.

A config is a YAML file. It may set `inherit: other.yaml` (resolved relative to
its own directory) to extend another config. CLI overrides use `key=value`
strings that are parsed as YAML scalars.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExpConfig:
    name: str = "unnamed"
    seed: int = 42

    # data
    data_dir: str = "data"
    out_dir: str = "runs"
    limit: int = 0                    # >0: subsample every split (smoke tests)
    max_len: int = 128

    # model
    model_name: str = "google/muril-base-cased"
    head: str = "cls"                 # cls | word | hier
    label_tree: bool = False          # OLID-style hierarchical label heads
    dropout: float = 0.1
    attn_dim: int = 256               # additive word-attention size

    # features
    lexicon: str = "off"              # off | curated | mined | both
    translit: str = "off"             # off | aug | tta | aug_tta

    # loss
    loss: str = "ce"                  # ce | focal
    class_weight: str = "sqrt"        # none | sqrt | balanced
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    # optimisation
    epochs: int = 5
    batch_size: int = 16              # per device
    grad_accum: int = 1
    lr: float = 2e-5
    head_lr_mult: float = 5.0
    llrd: float = 0.9                 # layer-wise LR decay (1.0 = off)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    patience: int = 2
    fp16: bool = True
    num_workers: int = 2

    # outputs
    keep_checkpoint: bool = False      # best.pt is ~1 GB for MuRIL-base; keep only where needed

    extra: dict = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name / f"seed{self.seed}"


def _load_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("inherit", None)
    if parent is None:
        return raw
    merged = _load_yaml(path.parent / parent)
    merged.update(raw)
    return merged


def load_config(path: str | None = None, overrides: list[str] | None = None) -> ExpConfig:
    values = _load_yaml(Path(path)) if path else {}
    for item in overrides or []:
        key, _, value = item.partition("=")
        values[key.strip()] = yaml.safe_load(value)

    fields = {f.name: f for f in dataclasses.fields(ExpConfig)}
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for key, value in values.items():
        default = fields[key].default
        # YAML 1.1 reads bare off/on as booleans (lexicon: off must stay a string)
        # and "1e-5" as a string (no decimal point), so coerce to the field's type.
        if isinstance(value, bool) and isinstance(default, str):
            values[key] = "on" if value else "off"
        elif isinstance(value, str) and isinstance(default, (int, float)) and not isinstance(default, bool):
            values[key] = type(default)(float(value))
    return ExpConfig(**values)


def save_config(cfg: ExpConfig, path: Path) -> None:
    path.write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
