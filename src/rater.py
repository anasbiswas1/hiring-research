"""The senior-responsibility rater.

One advertisement in, one word out (YES or NO), judged against the six-part
rubric of the paper (Table I) with the reading rules fixed after the human
adjudication of the validation set (rubric-v2). The model is the measuring instrument, so the
prompt text below is fixed and versioned, every call is cached on disk by
(model, prompt version, advertisement id, pass), and the model string the API
actually served is recorded with each verdict.

Sampling parameters cannot be set on the current API models, so determinism is
not available; stability is measured instead by rating every advertisement
twice and breaking ties with a third pass.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ads import OPENING_CHARS, opening

MODEL = "claude-sonnet-5-5"
PROMPT_VERSION = "rubric-v2"
MAX_TOKENS = 8

# Base prices per million tokens (input, output) used only for the spend projection.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5-5": (4.00, 20.00),
}

SYSTEM = (
    "You read job advertisements for a research study of entry-level software roles. "
    "You answer with exactly one word, YES or NO, and nothing else."
)

RUBRIC = (
    "1. Leadership and ownership: lead projects or own outcomes end to end.\n"
    "2. Mentorship: mentor, coach or train other engineers.\n"
    "3. Strategy: set technical strategy, direction or roadmap.\n"
    "4. Stakeholder influence: manage or influence stakeholders across teams.\n"
    "5. Architecture and design: design or architect systems, not only implement them.\n"
    "6. Autonomy: work independently with minimal supervision."
)

RULES = (
    "Rules for reading the six items:\n"
    "- Item 5 counts only when the advertisement makes the candidate responsible for the design or "
    "architecture of a system, solution or platform. A generic list of lifecycle verbs such as "
    "'design, develop, test and maintain' or 'design, analyze and build software systems' does not count.\n"
    "- Item 6 counts only when the advertisement says the candidate will work independently or with "
    "minimal supervision; 'self-motivated' or 'proactive' alone does not count.\n"
    "- Item 4 counts only when the candidate is asked to manage, influence or directly engage "
    "stakeholders, clients or business leaders; routine collaboration with teammates, product owners "
    "or testers does not count.\n"
    "- Items 1 to 3 count only when the advertisement assigns that responsibility to the candidate.\n"
    "- A description of what the company, the team or its engineers in general do is not a "
    "responsibility asked of the candidate. Years of experience or skills required are not responsibilities."
)

QUESTION = (
    "Does this advertisement ask the candidate to take on at least one responsibility "
    "ordinarily expected of more senior staff? Count only the six responsibilities below.\n\n"
    f"{RUBRIC}\n\n{RULES}\n\n"
    "Judge what the advertisement asks the candidate to do. Answer YES if the advertisement asks the "
    "candidate for at least one of the six responsibilities under these rules, otherwise answer NO.\n\n"
)


def user_message(title: str, desc: str, n_chars: int = OPENING_CHARS) -> str:
    return QUESTION + opening(title, desc, n_chars)


def thinking_setting(model: str):
    """Lowest thinking setting the model accepts; None means leave the API default."""
    if "sonnet-5-5" in model:
        return {"type": "between_tools"}
    if "haiku" in model:
        return {"type": "disabled"}
    return None


def request_kwargs(model: str, title: str, desc: str, n_chars: int = OPENING_CHARS) -> dict:
    kw = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": user_message(title, desc, n_chars)}],
    }
    th = thinking_setting(model)
    if th is not None:
        kw["thinking"] = th
    return kw


_WORD = re.compile(r"[A-Za-z]+")


def parse_verdict(text: str):
    """1 for a leading YES, 0 for a leading NO, None for anything else."""
    m = _WORD.search(text or "")
    if not m:
        return None
    w = m.group(0).upper()
    if w.startswith("YES") or w == "Y":
        return 1
    if w.startswith("NO") or w == "N":
        return 0
    return None


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_cost(ads: pd.DataFrame, passes: int, model: str = MODEL,
                  n_chars: int = OPENING_CHARS) -> dict:
    """Rough projection: about four characters per token, MAX_TOKENS output per call."""
    per_call_in = [estimate_tokens(SYSTEM) + estimate_tokens(user_message(t, d, n_chars))
                   for t, d in zip(ads.title, ads.desc)]
    calls = len(ads) * passes
    tok_in = sum(per_call_in) * passes
    tok_out = calls * MAX_TOKENS
    p_in, p_out = PRICES.get(model, (max(v[0] for v in PRICES.values()), max(v[1] for v in PRICES.values())))
    usd = tok_in / 1e6 * p_in + tok_out / 1e6 * p_out
    return {"calls": calls, "input_tokens": tok_in, "output_tokens_max": tok_out,
            "usd_projected": round(usd, 3), "model": model, "passes": passes}


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class RatingCache:
    """Append-only JSONL file; one line per API call."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        if self.path.exists():
            with self.path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self.rows.append(json.loads(line))

    def key(self, row) -> tuple:
        return (row["model"], row["prompt_version"], row["ad_id"], int(row["pass"]))

    def has(self, model, prompt_version, ad_id, pass_no) -> bool:
        k = (model, prompt_version, ad_id, int(pass_no))
        return any(self.key(r) == k for r in self.rows)

    def add(self, row: dict) -> None:
        self.rows.append(row)
        with self.path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    def frame(self, model=None, prompt_version=None) -> pd.DataFrame:
        rows = [r for r in self.rows
                if (model is None or r["model"] == model)
                and (prompt_version is None or r["prompt_version"] == prompt_version)]
        cols = ["model", "prompt_version", "ad_id", "pass", "verdict_raw", "verdict",
                "served_model", "input_tokens", "output_tokens", "stop_reason", "timestamp"]
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------- #
# API call with retries
# --------------------------------------------------------------------------- #
@dataclass
class CallResult:
    text: str
    served_model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


def call_once(client, kwargs: dict, retries: int = 5, base_sleep: float = 2.0) -> CallResult:
    last = None
    for attempt in range(retries):
        try:
            resp = client.messages.create(**kwargs)
            text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "text") == "text")
            usage = getattr(resp, "usage", None)
            return CallResult(
                text=text,
                served_model=getattr(resp, "model", kwargs["model"]),
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                stop_reason=str(getattr(resp, "stop_reason", "")),
            )
        except Exception as e:  # rate limits, 5xx, connection drops: wait and retry
            last = e
            status = getattr(e, "status_code", None)
            if status is not None and 400 <= int(status) < 500 and int(status) not in (408, 429):
                raise  # a request error will not fix itself by waiting
            time.sleep(base_sleep * (2 ** attempt))
    raise RuntimeError(f"API call failed after {retries} attempts: {last!r}")


def rate_pass(client, ads: pd.DataFrame, pass_no: int, cache: RatingCache,
              model: str = MODEL, prompt_version: str = PROMPT_VERSION,
              n_chars: int = OPENING_CHARS, limit: int | None = None,
              only_ad_ids=None, progress_every: int = 25) -> int:
    """Rate every advertisement in `ads` once for pass `pass_no`; skip cached rows.

    Returns the number of API calls made.
    """
    todo = ads
    if only_ad_ids is not None:
        todo = todo[todo.ad_id.isin(set(only_ad_ids))]
    todo = todo[[not cache.has(model, prompt_version, a, pass_no) for a in todo.ad_id]]
    if limit is not None:
        todo = todo.head(limit)
    n_calls = 0
    for i, r in enumerate(todo.itertuples(index=False), start=1):
        res = call_once(client, request_kwargs(model, r.title, r.desc, n_chars))
        cache.add({
            "model": model, "prompt_version": prompt_version, "ad_id": r.ad_id, "pass": int(pass_no),
            "verdict_raw": res.text.strip()[:40], "verdict": parse_verdict(res.text),
            "served_model": res.served_model, "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens, "stop_reason": res.stop_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        n_calls += 1
        if progress_every and i % progress_every == 0:
            print(f"  pass {pass_no}: {i}/{len(todo)} calls")
    return n_calls


def finalize(cache: RatingCache, ads: pd.DataFrame, model: str = MODEL,
             prompt_version: str = PROMPT_VERSION) -> pd.DataFrame:
    """One row per advertisement: pass1, pass2, pass3, final label and flags.

    final = majority of the non-missing passes; a two-pass tie needs pass 3; an
    advertisement with no parseable verdict at all is labelled 0 and flagged.
    """
    f = cache.frame(model, prompt_version)
    f = f[f.ad_id.isin(set(ads.ad_id))]
    wide = f.pivot_table(index="ad_id", columns="pass", values="verdict", aggfunc="first")
    wide = wide.reindex(columns=[1, 2, 3])
    wide.columns = ["pass1", "pass2", "pass3"]
    served = f.groupby("ad_id").served_model.agg(lambda s: ";".join(sorted(set(s))))
    tokens = f.groupby("ad_id")[["input_tokens", "output_tokens"]].sum()
    out = ads[["ad_id", "year", "source", "title"]].merge(wide, left_on="ad_id", right_index=True, how="left")
    out = out.merge(served.rename("served_model"), left_on="ad_id", right_index=True, how="left")
    out = out.merge(tokens, left_on="ad_id", right_index=True, how="left")

    def _final(row):
        votes = [v for v in (row.pass1, row.pass2, row.pass3) if pd.notna(v)]
        if not votes:
            return pd.Series({"final": 0, "n_votes": 0, "unstable": False, "malformed": True})
        yes = sum(int(v) for v in votes)
        no = len(votes) - yes
        if yes == no:  # only possible with two votes and no tie-break yet
            return pd.Series({"final": pd.NA, "n_votes": len(votes), "unstable": True, "malformed": False})
        return pd.Series({"final": int(yes > no), "n_votes": len(votes),
                          "unstable": bool(yes and no), "malformed": False})

    out = pd.concat([out, out.apply(_final, axis=1)], axis=1)
    return out


def needs_tie_break(final_df: pd.DataFrame) -> list:
    """Advertisements whose two passes disagree or are not both parseable."""
    f = final_df
    mask = (f.pass1.isna() | f.pass2.isna() | (f.pass1 != f.pass2)) & f.pass3.isna()
    return f.loc[mask, "ad_id"].tolist()


def stability(final_df: pd.DataFrame) -> dict:
    both = final_df.dropna(subset=["pass1", "pass2"])
    agree = float((both.pass1 == both.pass2).mean()) if len(both) else float("nan")
    return {"n_rated": int(len(final_df)), "n_with_two_passes": int(len(both)),
            "pass_agreement": agree, "n_tie_broken": int(final_df.pass3.notna().sum()),
            "n_malformed": int(final_df.malformed.sum())}


# --------------------------------------------------------------------------- #
# Offline stand-in for the API, for tests and dry runs. It is not a result source.
# --------------------------------------------------------------------------- #
_FAKE_CUES = re.compile(
    r"\b(lead|own|mentor|coach|train|strategy|roadmap|stakeholder|architect|design systems|"
    r"independently|minimal supervision|autonom)", re.I)


class _FakeBlock:
    def __init__(self, text):
        self.type, self.text = "text", text


class _FakeUsage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class _FakeResponse:
    def __init__(self, text, model, i):
        self.content = [_FakeBlock(text)]
        self.model = model + "-fake"
        self.usage = _FakeUsage(i, 1)
        self.stop_reason = "end_turn"


class _FakeMessages:
    def __init__(self, flaky_every=0):
        self.calls = 0
        self.flaky_every = flaky_every

    def create(self, **kw):
        self.calls += 1
        if self.flaky_every and self.calls % self.flaky_every == 0:
            raise ConnectionError("simulated transient failure")
        text = kw["messages"][0]["content"]
        body = text.split("Advertisement (opening):", 1)[-1]
        return _FakeResponse("YES" if _FAKE_CUES.search(body) else "NO", kw["model"], estimate_tokens(text))


class FakeClient:
    """Keyword stand-in with the same call shape as the real client."""

    def __init__(self, flaky_every=0):
        self.messages = _FakeMessages(flaky_every)
