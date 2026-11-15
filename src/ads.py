"""Advertisement records shared by every notebook.

An advertisement is a row with a title and a description. Its identifier is a
hash of the full text, so the same advertisement gets the same id in every run
and the rating cache can be keyed on it without storing the text in git.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

OPENING_CHARS = 700          # characters of the description shown to the rater (paper: first 500 to 700)
SUBSET_COLUMNS = ["ad_id", "year", "source", "title", "desc"]

_WS = re.compile(r"\s+")


def clean_text(text) -> str:
    """Collapse whitespace; keep the words unchanged."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return _WS.sub(" ", str(text)).strip()


def ad_id(title, desc) -> str:
    """Stable 16-hex identifier from the cleaned title and full description."""
    key = clean_text(title) + "\n" + clean_text(desc)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def opening(title, desc, n_chars: int = OPENING_CHARS) -> str:
    """The text the rater reads: the title and the first n_chars of the description."""
    return f"Title: {clean_text(title)}\n\nAdvertisement (opening): {clean_text(desc)[:n_chars]}"


def make_subset(df: pd.DataFrame, year: int, source: str,
                title_col: str = "title", desc_col: str = "desc") -> pd.DataFrame:
    """Standard subset frame: ad_id, year, source, title, desc (whitespace-cleaned)."""
    out = pd.DataFrame({
        "title": df[title_col].map(clean_text),
        "desc": df[desc_col].map(clean_text),
    })
    out = out[(out.title != "") & (out.desc != "")].copy()
    out.insert(0, "ad_id", [ad_id(t, d) for t, d in zip(out.title, out.desc)])
    out.insert(1, "year", int(year))
    out.insert(2, "source", source)
    out = out.drop_duplicates("ad_id").reset_index(drop=True)
    return out[SUBSET_COLUMNS]


def load_subsets(subset_dir: Path, years=(2019, 2024, 2026)) -> pd.DataFrame:
    """Concatenate data/subsets/junior_<year>.csv for the years that exist."""
    frames = []
    for y in years:
        p = Path(subset_dir) / f"junior_{y}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, dtype={"ad_id": str}))
    if not frames:
        raise FileNotFoundError(f"no junior_<year>.csv files under {subset_dir}")
    return pd.concat(frames, ignore_index=True)


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=_jsonable))


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except ImportError:
        pass
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not JSON serialisable: {type(x)}")
