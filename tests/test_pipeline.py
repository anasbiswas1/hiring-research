"""CPU checks of the pipeline plumbing. No network, no paid calls."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ads  # noqa: E402
import rater  # noqa: E402
import subsets  # noqa: E402
import validation  # noqa: E402
import results  # noqa: E402


def _toy_ads(n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        senior = i % 3 != 0
        body = ("You will lead the migration and mentor two interns. " if senior else "You will fix bugs with your team. ")
        body += " ".join(rng.choice(["alpha", "beta", "gamma", "delta"], size=40))
        rows.append({"title": f"Junior Software Engineer {i}", "desc": body})
    return ads.make_subset(pd.DataFrame(rows), 2024, "test")


def test_ad_id_is_stable_and_whitespace_insensitive():
    assert ads.ad_id("A  B", "x\n y") == ads.ad_id("A B", "x y")
    assert ads.ad_id("A", "x") != ads.ad_id("A", "y")


def test_parse_verdict():
    assert rater.parse_verdict("YES") == 1
    assert rater.parse_verdict(" no.") == 0
    assert rater.parse_verdict("Yes, because") == 1
    assert rater.parse_verdict("") is None
    assert rater.parse_verdict("maybe") is None


def test_request_shape_has_no_sampling_parameters():
    kw = rater.request_kwargs("claude-sonnet-5-5", "Junior Dev", "Build things.")
    assert "temperature" not in kw and "top_p" not in kw
    assert kw["thinking"] == {"type": "between_tools"}
    assert kw["max_tokens"] == rater.MAX_TOKENS
    kw2 = rater.request_kwargs("claude-haiku-5-5", "Junior Dev", "Build things.")
    assert kw2["thinking"] == {"type": "disabled"}
    assert "Mentorship" in kw["messages"][0]["content"]


def test_rating_cache_passes_and_tie_break(tmp_path):
    a = _toy_ads()
    cache = rater.RatingCache(tmp_path / "cache.jsonl")
    client = rater.FakeClient()
    n1 = rater.rate_pass(client, a, 1, cache, model="fake-model")
    n2 = rater.rate_pass(client, a, 2, cache, model="fake-model")
    assert n1 == len(a) and n2 == len(a)
    assert rater.rate_pass(client, a, 1, cache, model="fake-model") == 0  # cached, no new calls
    fin = rater.finalize(cache, a, model="fake-model")
    assert fin.final.notna().all() and rater.needs_tie_break(fin) == []
    assert fin.final.sum() == 8  # 2 of every 3 toy ads carry a cue
    st = rater.stability(fin)
    assert st["pass_agreement"] == 1.0 and st["n_tie_broken"] == 0
    reloaded = rater.RatingCache(tmp_path / "cache.jsonl")
    assert len(reloaded.rows) == 2 * len(a)


def test_retry_on_transient_failure(tmp_path):
    a = _toy_ads(n=4)
    cache = rater.RatingCache(tmp_path / "c.jsonl")
    client = rater.FakeClient(flaky_every=3)
    rater.call_once.__defaults__  # noqa: B018  (keep the signature visible in the test)
    n = rater.rate_pass(client, a, 1, cache, model="fake-model")
    assert n == 4 and len(cache.rows) == 4


def test_cost_projection_is_small():
    a = _toy_ads(n=100)
    est = rater.estimate_cost(a, passes=2, model="claude-sonnet-5-5")
    assert est["calls"] == 200 and est["usd_projected"] < 1.0


@pytest.mark.parametrize("title,keep", [
    ("Junior Software Engineer", True),
    ("Graduate Software Developer", True),
    ("Software Engineer I", True),
    ("Associate Application Developer", True),
    ("Jr. DevOps Engineer-NYC", True),
    ("2024 Launch Program: Junior Software Engineer Intern", True),
    ("Senior Software Engineer (Mobile)", False),
    ("Lead Software Engineer, Backend", False),
    ("Staff UI Software Engineer", False),
    ("Software Engineer III R-1833898", False),
    ("Junior Sales Engineer", False),
    ("Junior Software Engineer in Test", False),
    ("Business Development Associate", False),
    ("Software Engineer", False),            # no junior cue and no platform level
])
def test_subset_rules(title, keep):
    df = pd.DataFrame({"title": [title]})
    assert (len(subsets.select_junior_software(df)) == 1) == keep


def test_platform_level_counts_only_without_senior_cue():
    df = pd.DataFrame({"title": ["Software Engineer", "Senior Software Engineer", "Software Engineer"],
                       "formatted_experience_level": ["Entry level", "Entry level", "Mid-Senior level"]})
    out = subsets.select_junior_software(df, level_col="formatted_experience_level")
    assert list(out.title) == ["Software Engineer"] and list(out.why) == ["platform level"]
    counts = subsets.selection_counts(df, level_col="formatted_experience_level")
    assert counts["selected"] == 1 and counts["dropped_for_senior_cue"] == 1


def test_agreement_reproduces_paper_validation():
    # 40 advertisements: rater positive 17, all correct; human positive 22; 5 misses.
    human = [1] * 17 + [1] * 5 + [0] * 18
    rated = [1] * 17 + [0] * 5 + [0] * 18
    m = validation.agreement(pd.Series(human), pd.Series(rated))
    assert m["agreement"] == pytest.approx(0.875)
    assert m["kappa"] == pytest.approx(0.754, abs=0.001)
    assert m["precision"] == 1.0 and m["recall"] == pytest.approx(0.773, abs=0.001)


def test_sheet_roundtrip(tmp_path):
    a = _toy_ads(n=10)
    fin = a.assign(final=[1, 0] * 5)
    samp = validation.stratified_sample(fin, n_per_label=3, seed=1)
    assert len(samp) == 6 and set(samp.final) == {0, 1}
    sheet, key = validation.build_sheet({"test": samp}, a, n_chars=200, seed=1)
    assert "final" not in sheet.columns and len(sheet) == 6
    p = tmp_path / "sheet.xlsx"
    validation.write_sheet_xlsx(sheet, p)
    back = validation.read_sheet(p)
    assert list(back.sheet_id) == list(sheet.sheet_id) and back.human_senior.isna().all()


def test_results_table_and_figure(tmp_path):
    frames = []
    for y, n in [(2019, 20), (2024, 40), (2026, 30)]:
        a = _toy_ads(n=n, seed=y).assign(year=y)
        a["final"] = [(i % 3 != 0) for i in range(n)]
        a["why"] = ["title cue" if i % 2 else "platform level" for i in range(n)]
        if y == 2019:
            a["why"] = "title cue"
        frames.append(a)
    allads = pd.concat(frames, ignore_index=True)
    sizes = pd.DataFrame([{"year": 2024, "route": "title cue", "N": 300}, {"year": 2024, "route": "platform level", "N": 100},
                          {"year": 2026, "route": "title cue", "N": 15}, {"year": 2026, "route": "platform level", "N": 15},
                          {"year": 2019, "route": "title cue", "N": 20}])
    res = results.compute(allads[["ad_id", "year", "why", "final"]], allads, n_boot=200, seed=7, pool_sizes=sizes)
    assert (2019, "title cue") in res["strata"] and (2024, "platform level") in res["strata"]
    assert abs(res["years"][2024]["weights"]["title cue"] - 0.75) < 1e-9
    t2 = results.table2(res)
    assert "all routes, weighted by pool size" in set(t2["Selection"])
    results.fig2(res, tmp_path / "fig2.png", dpi=100)
    assert (tmp_path / "fig2.png").exists()
    s = results.summary(res)
    assert "2024|title cue" in s["strata"] and s["trend_title_cue"]["ok"]


def test_pick_column_and_parquet_schema(tmp_path):
    import fetch
    cols = ["uid", "Title", "description_core", "metro"]
    assert fetch.pick_column(cols, fetch.TITLE_CANDIDATES) == "Title"
    assert fetch.pick_column(cols, fetch.DESC_CANDIDATES) == "description_core"
    assert fetch.pick_column(cols, fetch.LEVEL_CANDIDATES, required=False) is None
    with pytest.raises(KeyError):
        fetch.pick_column(cols, fetch.LEVEL_CANDIDATES, what="level")
    pd.DataFrame({"title": ["a"], "description": ["b"]}).to_parquet(tmp_path / "x.parquet")
    assert fetch.parquet_columns(tmp_path / "x.parquet") == ["title", "description"]
