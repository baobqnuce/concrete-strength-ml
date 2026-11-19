"""Data download, loading and leakage-free splitting for the UCI Concrete dataset."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "concrete.csv"

# Official source (UCI ML Repository, id 165) is an .xls inside a zip; a CSV mirror
# with identical values (1030 rows, 9 columns) is used for convenience.
MIRROR_URL = (
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/"
    "master/concrete.csv"
)
EXPECTED_SHA256 = "d7d8bd087f832935e902bcb2687667238cac3fe06799677a93ca2f46dce8db02"

INGREDIENTS = ["cement", "slag", "ash", "water", "superplastic", "coarseagg", "fineagg"]
FEATURES = INGREDIENTS + ["age"]
TARGET = "strength"
SEED = 42


def download(path: Path = DATA_PATH) -> Path:
    """Download the CSV mirror once and verify its checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {MIRROR_URL} -> {path}")
        urllib.request.urlretrieve(MIRROR_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Checksum mismatch for {path}: {digest}")
    return path


def load() -> pd.DataFrame:
    df = pd.read_csv(download())
    assert list(df.columns) == FEATURES + [TARGET], df.columns
    assert len(df) == 1030
    return df


def mix_groups(df: pd.DataFrame) -> np.ndarray:
    """One integer id per unique mix design (same ingredients, any age).

    The same mix is tested at several ages (e.g. 3, 7, 28, 90 days). If those rows
    are spread over train and test, the test score is optimistic (data leakage).
    """
    key = df[INGREDIENTS].round(2).astype(str).agg("|".join, axis=1)
    return pd.factorize(key)[0]


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED):
    """Group-aware development/test split. Returns (dev_df, test_df)."""
    groups = mix_groups(df)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev_idx, test_idx = next(gss.split(df, groups=groups))
    return df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
