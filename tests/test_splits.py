"""Leakage checks on the released split files. Run:  pytest -q   (put the CSVs in configs/splits/ first)."""
import os, pandas as pd, pytest

SPLITS = os.path.join(os.path.dirname(__file__), "..", "configs", "splits")

def _load(name):
    f = os.path.join(SPLITS, name)
    if not os.path.exists(f): pytest.skip(f"{name} not found")
    return pd.read_csv(f)

def test_paddy_groups_do_not_straddle_splits():
    df = _load("paddy_split.csv")
    assert (df.groupby("group").split.nunique() == 1).all()

def test_paddy_all_classes_in_every_split():
    df = _load("paddy_split.csv")
    assert df.groupby("split").label.nunique().eq(df.label.nunique()).all()

def test_no_path_in_two_splits():
    for name in ("paddy_split.csv", "plantdoc_split.csv"):
        df = _load(name); assert df.path.is_unique

def test_plantdoc_official_test_untouched():
    df = _load("plantdoc_split.csv")
    assert (df[df.orig_split == "test"].split == "test").all()
    assert df.label.nunique() == 27
