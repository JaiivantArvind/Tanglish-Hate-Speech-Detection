"""Loading the DravidianCodeMix Tamil offensive-language splits (6 classes)."""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATASET_URL = (
    "https://raw.githubusercontent.com/bharathichezhiyan/"
    "DravidianCodeMix-Dataset/master/DravidianCodeMix-2020.zip"
)

# Order matters: ids are used in saved logits. "Untargetede" is the dataset's own spelling.
LABELS = [
    "Not_offensive",
    "Offensive_Untargetede",
    "Offensive_Targeted_Insult_Individual",
    "Offensive_Targeted_Insult_Group",
    "Offensive_Targeted_Insult_Other",
    "not-Tamil",
]
SHORT_LABELS = ["NOT", "UNT", "IND", "GRP", "OTH", "NTA"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
SPLITS = ("train", "dev", "test")


def download_dataset(data_dir: str | Path) -> Path:
    """Downloads and extracts the dataset zip if the Tamil files are not present."""
    folder = Path(data_dir) / "DravidianCodeMix"
    if (folder / "tamil_offensive_full_train.csv").exists():
        return folder
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_URL} ...")
    with urllib.request.urlopen(DATASET_URL) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(data_dir)
    return folder


def text_key(text: str) -> str:
    """Normalised key used to detect duplicates and train/test overlap."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def read_split(path: Path) -> pd.DataFrame:
    # Files are "text<TAB>label<TAB>" with no header and no quoting.
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rows.append((parts[0], parts[1].strip()))
    df = pd.DataFrame(rows, columns=["text", "category"])
    unknown = set(df["category"]) - set(LABELS)
    if unknown:
        raise ValueError(f"{path.name}: unexpected labels {unknown}")
    df["label"] = df["category"].map(LABEL2ID).astype(int)
    return df


def load_splits(data_dir: str | Path, clean: bool = True, verbose: bool = False) -> dict[str, pd.DataFrame]:
    """Returns {"train", "dev", "test"} DataFrames with columns text, category, label.

    With clean=True the training split is de-duplicated (majority label wins) and
    any training text that also occurs in dev or test is removed. Dev and test are
    never modified, so scores stay comparable with the shared task.
    """
    folder = download_dataset(data_dir)
    splits = {s: read_split(folder / f"tamil_offensive_full_{s}.csv") for s in SPLITS}
    if not clean:
        return splits

    train = splits["train"].copy()
    train["key"] = train["text"].map(text_key)
    n0 = len(train)

    majority = train.groupby("key")["label"].agg(lambda s: s.value_counts().idxmax())
    train = train.drop_duplicates("key").copy()
    train["label"] = train["key"].map(majority)
    train["category"] = train["label"].map(dict(enumerate(LABELS)))
    n_dedup = len(train)

    held_out = set(splits["dev"]["text"].map(text_key)) | set(splits["test"]["text"].map(text_key))
    train = train[~train["key"].isin(held_out)]

    if verbose:
        print(f"train: {n0} rows -> {n_dedup} after de-duplication -> {len(train)} after removing dev/test overlap")
    splits["train"] = train.drop(columns="key").reset_index(drop=True)
    return splits


def subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified-ish subsample that keeps at least one row of every class present."""
    if n <= 0 or n >= len(df):
        return df
    first = df.groupby("label", group_keys=False).head(1)
    rest = df.drop(first.index).sample(n=max(0, n - len(first)), random_state=seed)
    return pd.concat([first, rest]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
