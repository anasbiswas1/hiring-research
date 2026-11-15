"""Junior software subset rules, written out so a reader can audit every selection.

The paper's rule, in words: keep software-development titles (software engineer
or developer, full-stack, back-end, front-end, web, mobile and application
developers, DevOps and platform engineers, programmer-analysts), drop
non-software roles that share the vocabulary, and take junior status from title
cues (junior, graduate, entry-level, new grad, trainee, intern, associate) or
from the platform's seniority field where one exists.

One tightening is applied everywhere: a title that carries an explicit senior
cue (senior, lead, staff, principal, manager, head, director, architect,
supervisor) is never a junior advertisement, whatever the platform field says.
Platform seniority fields are filled in by the poster and are not reliable.
"""
from __future__ import annotations

import re

import pandas as pd

SOFTWARE = re.compile(
    r"(?<![a-z0-9])(software\s+(engineer|developer|development|engineering)s?"
    r"|(full[\s-]?stack|back[\s-]?end|front[\s-]?end|web|mobile|ios|android|application|applications|app)"
    r"\s+(engineer|developer)s?"
    r"|(fullstack|backend|frontend)\s*(engineer|developer)?s?"
    r"|dev\s?ops(\s+(engineer|developer))?"
    r"|platform\s+engineer"
    r"|site\s+reliability\s+engineer"
    r"|programmer[\s/-]*analyst"
    r"|\bsde\b|\bswe\b"
    r"|(java|javascript|typescript|python|\.net|dotnet|c#|c\+\+|php|react|node|angular|golang|ruby|scala|kotlin|swift)"
    r"\s+(developer|engineer)s?"
    r"|developer)(?![a-z0-9])", re.I)

# Roles that share the vocabulary but are not software development.
EXCLUDE = re.compile(
    r"\b(sales\s+engineer|hardware|electrical|mechanical|civil|structural|chemical|process\s+engineer"
    r"|network\s+engineer|field\s+engineer|service\s+engineer|support\s+engineer|customer\s+engineer"
    r"|solutions?\s+engineer|pre[\s-]?sales|account|recruit|business\s+develop\w*"
    r"|data\s+scientist|data\s+analyst|business\s+analyst|financial\s+analyst|marketing"
    r"|qa\s+engineer|qa\s*&|test\s+engineer|quality\s+engineer|sdet|in\s+test\b|tester"
    r"|project\s+manager|product\s+manager|program\s+manager|scrum\s+master"
    r"|technician|administrator|admin\b|help\s*desk|desktop\s+support"
    r"|real\s+estate|property\s+developer|land\s+developer|game\s+designer|graphic\s+designer)\b", re.I)

JUNIOR = re.compile(
    r"\b(junior|jr\.?|graduate|grad|entry[\s-]?level|new\s+grad|early\s+career|trainee|intern|internship"
    r"|associate|apprentice|fresher|trainee\w*|campus|university\s+hire"
    r"|(engineer|developer|engineering)\s+(i|1)(?![a-z0-9]))\b", re.I)

# Explicit non-junior cues: a title carrying one is never a junior advertisement.
SENIOR = re.compile(
    r"\b(senior|sr\.?|lead|staff|principal|manager|mgr\.?|head|director|architect|supervisor|chief"
    r"|vp|vice\s+president|team\s*lead|teamlead|mid[\s-]?level|iii|iv|v)\b", re.I)

# LinkedIn's formatted_experience_level values that count as junior.
JUNIOR_LEVELS = {"entry level", "internship", "associate"}


def is_software(title: str) -> bool:
    t = str(title)
    return bool(SOFTWARE.search(t)) and not EXCLUDE.search(t)


def has_junior_cue(title: str) -> bool:
    return bool(JUNIOR.search(str(title)))


def has_senior_cue(title: str) -> bool:
    return bool(SENIOR.search(str(title)))


def level_is_junior(level) -> bool:
    if level is None or (isinstance(level, float) and pd.isna(level)):
        return False
    return str(level).strip().lower() in JUNIOR_LEVELS


def select_junior_software(df: pd.DataFrame, title_col: str = "title",
                           level_col: str | None = None) -> pd.DataFrame:
    """Return the rows that pass, with a `why` column (title cue, platform level or both)."""
    t = df[title_col].astype(str)
    soft = t.map(is_software)
    jun = t.map(has_junior_cue)
    sen = t.map(has_senior_cue)
    lvl = df[level_col].map(level_is_junior) if level_col and level_col in df.columns else pd.Series(False, index=df.index)
    keep = soft & ~sen & (jun | lvl)
    out = df[keep].copy()
    why = []
    for j, l in zip(jun[keep], lvl[keep]):
        why.append("title cue and platform level" if (j and l) else ("title cue" if j else "platform level"))
    out["why"] = why
    return out


def selection_counts(df: pd.DataFrame, title_col: str = "title", level_col: str | None = None) -> dict:
    """How many rows each rule removes, for the subset report."""
    t = df[title_col].astype(str)
    soft = t.map(is_software)
    jun = t.map(has_junior_cue)
    sen = t.map(has_senior_cue)
    lvl = df[level_col].map(level_is_junior) if level_col and level_col in df.columns else pd.Series(False, index=df.index)
    return {
        "rows": int(len(df)),
        "software_title": int(soft.sum()),
        "software_and_junior_cue": int((soft & jun).sum()),
        "software_and_junior_level": int((soft & lvl).sum()),
        "software_and_junior_either": int((soft & (jun | lvl)).sum()),
        "dropped_for_senior_cue": int((soft & (jun | lvl) & sen).sum()),
        "selected": int((soft & ~sen & (jun | lvl)).sum()),
    }
