"""Downloads and schema helpers for the raw corpora (Zenodo record files, parquet columns)."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records/{record}"


def zenodo_files(record_id: str) -> list[dict]:
    """[{key, size, checksum, url}] for every file of a public Zenodo record."""
    with urllib.request.urlopen(ZENODO_API.format(record=record_id), timeout=60) as r:
        meta = json.loads(r.read().decode("utf-8"))
    out = []
    for f in meta.get("files", []):
        url = (f.get("links") or {}).get("self") or f"https://zenodo.org/records/{record_id}/files/{f['key']}?download=1"
        out.append({"key": f["key"], "size": int(f.get("size", 0)), "checksum": f.get("checksum", ""), "url": url})
    return out


def download(url: str, path: Path, expected_size: int | None = None, chunk: int = 8 << 20) -> Path:
    """Stream a file to disk; skip when it is already there with the expected size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        print(f"present: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    done = 0
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as fh:
        while True:
            block = r.read(chunk)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if expected_size:
                print(f"\r  {path.name}: {100 * done / expected_size:5.1f}%", end="")
    print()
    if expected_size and done != expected_size:
        raise IOError(f"{path.name}: got {done} bytes, expected {expected_size}")
    tmp.replace(path)
    return path


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq
    return pq.read_schema(str(path)).names


def pick_column(columns: list[str], candidates: list[str], required: bool = True, what: str = "") -> str | None:
    """First candidate present in `columns` (case-insensitive); error with the schema if none and required."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise KeyError(f"no {what or 'column'} among {candidates} in schema {columns}: set the column name in SETTINGS")
    return None


TITLE_CANDIDATES = ["title", "job_title", "jobtitle", "position"]
DESC_CANDIDATES = ["description", "job_description", "description_core", "description_raw", "desc", "text", "job_summary"]
LEVEL_CANDIDATES = ["formatted_experience_level", "experience_level", "seniority_level", "job_level", "level"]
DATE_CANDIDATES = ["posted_at", "date_posted", "posted_date", "listed_time", "original_listed_time", "first_seen", "first_seen_at", "scraped_at", "observed_at"]
LOCATION_CANDIDATES = ["location", "metro", "city", "job_location"]
ID_CANDIDATES = ["uid", "job_id", "id", "posting_id", "job_posting_id"]
