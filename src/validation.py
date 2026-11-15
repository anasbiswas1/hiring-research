"""Human validation of the rater: the blind labelling sheet and the agreement statistics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SHEET_COLUMNS = ["sheet_id", "source", "title", "opening", "human_senior", "responsibilities", "note"]
RUBRIC_CODES = "L = leadership/ownership, M = mentorship, S = strategy, K = stakeholder influence, A = architecture/design, U = autonomy"


def stratified_sample(final_df: pd.DataFrame, n_per_label: int, seed: int,
                      label_col: str = "final") -> pd.DataFrame:
    """n_per_label advertisements rated 1 and n_per_label rated 0, drawn at random."""
    rng = np.random.default_rng(seed)
    parts = []
    for lab in (1, 0):
        pool = final_df[final_df[label_col] == lab]
        k = min(n_per_label, len(pool))
        parts.append(pool.iloc[rng.choice(len(pool), size=k, replace=False)])
    out = pd.concat(parts)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def build_sheet(blocks: dict, ads: pd.DataFrame, n_chars: int, seed: int) -> pd.DataFrame:
    """blocks: {source_label: DataFrame with ad_id}. Rater columns are left out so the labelling is blind.

    Rows are shuffled within each block; the mapping sheet_id -> ad_id is returned separately.
    """
    from ads import clean_text
    rng = np.random.default_rng(seed)
    rows, key = [], []
    for src, block in blocks.items():
        ids = list(block.ad_id)
        rng.shuffle(ids)
        lookup = ads.set_index("ad_id")
        for i, aid in enumerate(ids, start=1):
            sid = f"{src}-{i:03d}"
            rows.append({"sheet_id": sid, "source": src, "title": lookup.loc[aid, "title"],
                         "opening": clean_text(lookup.loc[aid, "desc"])[:n_chars],
                         "human_senior": "", "responsibilities": "", "note": ""})
            key.append({"sheet_id": sid, "ad_id": aid})
    return pd.DataFrame(rows, columns=SHEET_COLUMNS), pd.DataFrame(key)


def write_sheet_xlsx(sheet: pd.DataFrame, path: Path) -> None:
    """Excel file with wrapped text and a frozen header; the labeller fills human_senior with 1 or 0."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    ws.append(list(sheet.columns))
    for r in sheet.itertuples(index=False):
        ws.append(list(r))
    widths = {"sheet_id": 14, "source": 10, "title": 36, "opening": 110, "human_senior": 13, "responsibilities": 16, "note": 30}
    for i, c in enumerate(sheet.columns, start=1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(c, 14)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    guide = wb.create_sheet("how_to_label")
    lines = [
        "Read the title and the opening of each advertisement, then put 1 in human_senior if it asks the candidate",
        "for at least one of the six responsibilities below, otherwise 0. Do not look anything up and do not skip rows.",
        "",
        "1. Leadership and ownership: lead projects or own outcomes end to end.",
        "2. Mentorship: mentor, coach or train other engineers.",
        "3. Strategy: set technical strategy, direction or roadmap.",
        "4. Stakeholder influence: manage or influence stakeholders across teams.",
        "5. Architecture and design: design or architect systems, not only implement them.",
        "6. Autonomy: work independently with minimal supervision.",
        "",
        "Optional: in responsibilities, write the letters that apply (" + RUBRIC_CODES + ").",
        "Judge what the advertisement asks the candidate to do, not what the company or the team does.",
    ]
    for ln in lines:
        guide.append([ln])
    guide.column_dimensions["A"].width = 120
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def read_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="labels", dtype=str)
    df["human_senior"] = pd.to_numeric(df["human_senior"], errors="coerce")
    return df


def agreement(human: pd.Series, rater: pd.Series) -> dict:
    """Agreement, Cohen's kappa, precision and recall of the rater against the human labels."""
    h = pd.Series(human).astype(int).to_numpy()
    r = pd.Series(rater).astype(int).to_numpy()
    n = len(h)
    tp = int(((h == 1) & (r == 1)).sum())
    fp = int(((h == 0) & (r == 1)).sum())
    fn = int(((h == 1) & (r == 0)).sum())
    tn = int(((h == 0) & (r == 0)).sum())
    po = (tp + tn) / n
    pe = ((tp + fp) / n) * ((tp + fn) / n) + ((fn + tn) / n) * ((fp + tn) / n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"n": n, "agreement": po, "kappa": kappa,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "human_positive": tp + fn, "rater_positive": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
