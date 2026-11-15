"""
dedup_inference.py
==================
Template-aware (duplicate-aware) inference for the junior-software
seniorization analysis.

WHY THIS EXISTS
---------------
Reviewers correctly noted that the 2019 corpus is dominated by near-identical
advertisements from a single staffing firm. A template repeated dozens of times
is NOT dozens of independent observations, so an ordinary bootstrap / chi-square
computed on raw rows understates uncertainty (the confidence interval is
"optimistically narrow") and can manufacture apparent stability.

This module:
  1. detects near-duplicate template clusters with MinHash + LSH,
  2. reports how much duplication there is (per year),
  3. recomputes the senior-associated prevalence with a CLUSTER bootstrap
     (resampling templates, not rows) and reports the design effect and the
     effective sample size,
  4. gives a one-observation-per-template ("collapsed") estimate as the most
     conservative reading, and
  5. fits a duplicate-robust logistic trend across years with cluster-robust
     standard errors.

It does NOT fix the source==year confound (Monster 2019 vs LinkedIn 2024/26).
Clustering corrects the *variance*; it cannot separate platform from time. Only
a single-source junior panel can do that. This is stated in the printed report.

INPUT CONTRACT
--------------
A pandas DataFrame `df` with (at least) these columns:
  - text   : str   the advertisement text actually shown to the classifier
                   (title + opening section is fine; must be what was rated)
  - year   : int   e.g. 2019, 2024, 2026
  - senior : int   the binary classifier label (1 = senior-associated, 0 = not)
An optional `job_id` column is passed through if present.

USAGE (in Colab, on your real subsets)
--------------------------------------
    from dedup_inference import analyze
    results = analyze(df, text_col="text", year_col="year", senior_col="senior",
                      threshold=0.85, n_boot=5000, seed=42)// results is a dict; the function also prints a full report.

Run this file directly (`python dedup_inference.py`) to see it work on a
synthetic corpus with planted templates.
"""

from __future__ import annotations
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH


# --------------------------------------------------------------------------- #
# 1. Near-duplicate template detection
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    """Lowercase, strip URLs/emails/digits noise, collapse whitespace."""
    t = str(text).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\S+@\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)        # keep letters + space only
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _shingles(text: str, k: int = 5) -> set[str]:
    """Word k-shingles. k=5 is a good default for near-duplicate ad text."""
    words = _normalize(text).split()
    if len(words) < k:
        # short text: fall back to the whole thing as one shingle
        return {" ".join(words)} if words else {""}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _minhash(text: str, num_perm: int, k: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in _shingles(text, k):
        m.update(sh.encode("utf8"))
    return m


class _UnionFind:
    """Dependency-free connected components."""
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cluster_templates(texts, threshold: float = 0.85,
                      num_perm: int = 128, k: int = 5) -> np.ndarray:
    """
    Return an integer cluster label per text. Texts whose estimated Jaccard
    similarity exceeds `threshold` are unioned into the same template cluster.
    Singletons get their own cluster id.
    """
    texts = list(texts)
    n = len(texts)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mhs = []
    for i, t in enumerate(texts):
        m = _minhash(t, num_perm, k)
        mhs.append(m)
        lsh.insert(str(i), m)

    uf = _UnionFind(n)
    for i in range(n):
        for j in lsh.query(mhs[i]):
            j = int(j)
            if j != i:
                uf.union(i, j)

    # relabel roots to compact 0..K-1
    roots = [uf.find(i) for i in range(n)]
    remap = {r: c for c, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots], dtype=int)


# --------------------------------------------------------------------------- #
# 2. Duplication report
# --------------------------------------------------------------------------- #
def duplication_report(df: pd.DataFrame, cluster_col="cluster",
                       year_col="year") -> pd.DataFrame:
    rows = []
    for yr, g in df.groupby(year_col):
        n = len(g)
        k = g[cluster_col].nunique()
        sizes = g[cluster_col].value_counts()
        rows.append({
            "year": yr,
            "n_ads": n,
            "n_templates": k,
            "dup_rate": round(1 - k / n, 3) if n else np.nan,
            "largest_template": int(sizes.iloc[0]) if len(sizes) else 0,
            "top_template_share": round(sizes.iloc[0] / n, 3) if n else np.nan,
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Cluster bootstrap prevalence (+ naive comparison + design effect)
# --------------------------------------------------------------------------- #
def _naive_boot(senior, n_boot, rng):
    senior = np.asarray(senior)
    n = len(senior)
    return np.array([senior[rng.integers(0, n, n)].mean() for _ in range(n_boot)])


def _cluster_boot(senior, cluster, n_boot, rng):
    """Resample whole templates with replacement; pool their rows."""
    senior = np.asarray(senior)
    cluster = np.asarray(cluster)
    by_cluster = {c: senior[cluster == c] for c in np.unique(cluster)}
    clusters = np.array(list(by_cluster.keys()))
    K = len(clusters)
    out = np.empty(n_boot)
    for b in range(n_boot):
        drawn = clusters[rng.integers(0, K, K)]
        vals = np.concatenate([by_cluster[c] for c in drawn])
        out[b] = vals.mean()
    return out


def prevalence_with_ci(df: pd.DataFrame, senior_col="senior",
                       cluster_col="cluster", n_boot=5000, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    senior = df[senior_col].to_numpy()
    cluster = df[cluster_col].to_numpy()
    point = senior.mean()

    naive = _naive_boot(senior, n_boot, rng)
    clust = _cluster_boot(senior, cluster, n_boot, rng)

    naive_ci = np.percentile(naive, [2.5, 97.5])
    clust_ci = np.percentile(clust, [2.5, 97.5])

    # design effect = variance inflation from clustering; effective n
    deff = np.var(clust, ddof=1) / max(np.var(naive, ddof=1), 1e-12)
    n_ads = len(senior)
    n_eff = n_ads / deff if deff > 0 else np.nan

    # collapsed: one observation per template (modal label within template)
    modal = (df.groupby(cluster_col)[senior_col].mean() >= 0.5).astype(int)
    collapsed_point = modal.mean()
    cm = modal.to_numpy()
    Kc = len(cm)
    collapsed_boot = np.array([cm[rng.integers(0, Kc, Kc)].mean()
                               for _ in range(n_boot)])
    collapsed_ci = np.percentile(collapsed_boot, [2.5, 97.5])

    return {
        "n_ads": int(n_ads),
        "n_templates": int(df[cluster_col].nunique()),
        "point": float(point),
        "naive_ci": (float(naive_ci[0]), float(naive_ci[1])),
        "cluster_ci": (float(clust_ci[0]), float(clust_ci[1])),
        "design_effect": float(deff),
        "effective_n": float(n_eff),
        "collapsed_point": float(collapsed_point),
        "collapsed_ci": (float(collapsed_ci[0]), float(collapsed_ci[1])),
    }


# --------------------------------------------------------------------------- #
# 4. Duplicate-robust logistic trend across years
# --------------------------------------------------------------------------- #
def clustered_trend(df: pd.DataFrame, senior_col="senior", year_col="year",
                    cluster_col="cluster") -> dict:
    """
    Logistic regression of senior ~ year (year centred, as a linear term)
    with cluster-robust standard errors (clusters = templates).
    Returns the per-year odds ratio and its clustered 95% CI. If statsmodels
    is unavailable or the model fails to converge, returns {'ok': False}.
    """
    try:
        import statsmodels.api as sm
    except Exception as e:  # pragma: no cover
        return {"ok": False, "reason": f"statsmodels unavailable: {e}"}

    d = df[[senior_col, year_col, cluster_col]].dropna().copy()
    yrs = np.sort(d[year_col].unique())
    if len(yrs) < 2:
        return {"ok": False, "reason": "need >=2 years for a trend"}

    # centre year on its mean so the intercept is interpretable
    d["year_c"] = d[year_col] - d[year_col].mean()
    X = sm.add_constant(d[["year_c"]].astype(float))
    y = d[senior_col].astype(float)
    try:
        model = sm.GLM(y, X, family=sm.families.Binomial())
        res = model.fit(cov_type="cluster",
                        cov_kwds={"groups": d[cluster_col].to_numpy()})
        beta = res.params["year_c"]
        ci = res.conf_int().loc["year_c"]
        return {
            "ok": True,
            "or_per_year": float(np.exp(beta)),
            "or_ci": (float(np.exp(ci[0])), float(np.exp(ci[1]))),
            "p_value": float(res.pvalues["year_c"]),
            "note": "OR per calendar year for senior-associated label; "
                    "clusters = templates.",
        }
    except Exception as e:  # pragma: no cover
        return {"ok": False, "reason": str(e)}


# --------------------------------------------------------------------------- #
# 5. Orchestration + human-readable report
# --------------------------------------------------------------------------- #
def analyze(df: pd.DataFrame, text_col="text", year_col="year",
            senior_col="senior", threshold=0.85, n_boot=5000, seed=42,
            num_perm=128, shingle_k=5) -> dict:
    df = df.copy()
    for c in (text_col, year_col, senior_col):
        if c not in df.columns:
            raise ValueError(f"missing required column: {c!r}")

    # cluster across the WHOLE pool so cross-year duplicates are caught too
    df["cluster"] = cluster_templates(df[text_col], threshold=threshold,
                                      num_perm=num_perm, k=shingle_k)

    dup = duplication_report(df, "cluster", year_col)

    per_year = {}
    for yr, g in df.groupby(year_col):
        per_year[int(yr)] = prevalence_with_ci(
            g, senior_col=senior_col, cluster_col="cluster",
            n_boot=n_boot, seed=seed)

    pooled = prevalence_with_ci(df, senior_col=senior_col,
                                cluster_col="cluster", n_boot=n_boot, seed=seed)
    trend = clustered_trend(df, senior_col, year_col, "cluster")

    _print_report(dup, per_year, trend)
    return {"duplication": dup, "per_year": per_year,
            "pooled": pooled, "trend": trend, "df": df}


def _fmt_ci(ci):  # 0-1 floats -> percent
    return f"[{100*ci[0]:.1f}, {100*ci[1]:.1f}]"


def _print_report(dup, per_year, trend):
    line = "=" * 74
    print(line)
    print("DUPLICATE-AWARE SENIORIZATION INFERENCE")
    print(line)
    print("\n[1] Template duplication by year")
    print(dup.to_string(index=False))
    print("\n    dup_rate = 1 - templates/ads ; top_template_share = largest "
          "single template\n    as a fraction of that year's ads.")

    print("\n[2] Prevalence: naive vs template-aware (per year)")
    print(f"    {'year':>6} {'ads':>5} {'tmpl':>5} {'point':>7} "
          f"{'naive 95% CI':>16} {'cluster 95% CI':>18} {'DEFF':>6} {'n_eff':>7}")
    for yr, r in sorted(per_year.items()):
        print(f"    {yr:>6} {r['n_ads']:>5} {r['n_templates']:>5} "
              f"{100*r['point']:>6.1f}% {_fmt_ci(r['naive_ci']):>16} "
              f"{_fmt_ci(r['cluster_ci']):>18} {r['design_effect']:>6.2f} "
              f"{r['effective_n']:>7.1f}")
    print("\n    The cluster CI is the honest one. When it is much wider than")
    print("    the naive CI (DEFF >> 1, n_eff << ads), template duplication was")
    print("    inflating precision. Collapsed (one-per-template) estimates:")
    for yr, r in sorted(per_year.items()):
        print(f"      {yr}: {100*r['collapsed_point']:.1f}% "
              f"{_fmt_ci(r['collapsed_ci'])}  (n_templates="
              f"{r['n_templates']})")

    print("\n[3] Duplicate-robust trend across years (cluster-robust logit)")
    if trend.get("ok"):
        print(f"    OR per year = {trend['or_per_year']:.3f}  "
              f"95% CI {_fmt_or(trend['or_ci'])}  p = {trend['p_value']:.3f}")
        print("    OR ~ 1 with a CI spanning 1 => no detectable trend once")
        print("    template correlation is accounted for.")
    else:
        print(f"    trend not estimated: {trend.get('reason')}")

    print("\n[4] Caveat this pipeline does NOT resolve")
    print("    Year is confounded with data source (e.g. Monster 2019 vs")
    print("    LinkedIn 2024/26). Clustering corrects the variance, not this")
    print("    confound. A single-source junior panel is required to separate")
    print("    platform from time; report the trend as source-confounded.")
    print(line)


def _fmt_or(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


# --------------------------------------------------------------------------- #
# 6. Synthetic demonstration (planted templates)
# --------------------------------------------------------------------------- #
_INTRO = ["We are hiring a", "Join our team as a", "Exciting opportunity for a",
          "Our client seeks a", "Growing scale-up looking for a",
          "Fintech company hiring a", "Remote-first team wants a"]
_ROLE = ["Junior Software Engineer", "Graduate Developer", "Entry-Level Backend Engineer",
         "Associate Full-Stack Developer", "Junior Frontend Engineer",
         "Graduate Platform Engineer", "Trainee Software Developer"]
_SENIOR_RESP = [
    "you will design and architect scalable distributed systems",
    "own the full delivery lifecycle from design through production",
    "mentor and coach other engineers on the team",
    "set the technical roadmap for your product area",
    "manage stakeholder relationships across product and design",
    "work independently with minimal supervision on ambiguous problems",
    "lead cross-functional projects end to end"]
_JUNIOR_RESP = [
    "complete well-defined coding tasks under close supervision",
    "fix simple bugs and write unit tests with guidance",
    "pair with senior engineers to learn our codebase",
    "assist the team with clearly scoped tickets",
    "follow established patterns on maintenance work",
    "contribute to documentation and small features"]
_TECH = ["Python and Django", "TypeScript and React", "Java and Spring",
         "Go microservices", "AWS and Kubernetes", "Ruby on Rails",
         "Node.js and GraphQL", "C# and .NET"]
_PERK = ["Competitive salary and equity.", "Hybrid working offered.",
         "Great learning budget.", "Inclusive, supportive culture.",
         "25 days holiday plus bank holidays.", "Flexible hours."]
_CITY = ["London", "Manchester", "Berlin", "Paris", "Austin", "Remote (EU)",
         "Sydney", "Toronto", "Dublin", "Bristol"]


def _make_ad(rng, senior: int) -> str:
    """A genuinely varied advertisement built from randomized components."""
    resp_pool = _SENIOR_RESP if senior else _JUNIOR_RESP
    k = int(rng.integers(2, 4))
    resp = ", ".join(rng.choice(resp_pool, size=k, replace=False))
    return (f"{rng.choice(_INTRO)} {rng.choice(_ROLE)} in {rng.choice(_CITY)}. "
            f"In this role {resp}. You will work with {rng.choice(_TECH)}. "
            f"{rng.choice(_PERK)} Ref {rng.integers(10000, 99999)}.")


def _synthetic_corpus(seed=0) -> pd.DataFrame:
    """
    Mimic the paper's problem: a 2019 year dominated by ONE staffing-firm
    template repeated ~30 times, with genuine variety everywhere else.
    Target prevalences: 2019 ~83%, 2024 ~80%, 2026 ~81%.
    """
    rng = np.random.default_rng(seed)
    rows = []

    # the planted staffing template: ONE fixed senior-flavoured ad, repeated
    template = ("Software Engineer wanted by leading staffing partner. You will "
                "design and architect scalable systems, own delivery end to end, "
                "mentor peers, and work independently with minimal supervision. "
                "Apply now to join our national talent network.")
    for _ in range(30):
        # tiny per-posting jitter (a ref code) as happens in real boards
        rows.append((template + f" Position {rng.integers(1000,9999)}", 2019, 1))

    # 2019 remainder: 17 genuinely varied ads (~65% senior so year avg ~83%)
    for _ in range(17):
        s = int(rng.random() < 0.65)
        rows.append((_make_ad(rng, s), 2019, s))

    # 2024: 120 varied ads (~80% senior-associated)
    for _ in range(120):
        s = int(rng.random() < 0.80)
        rows.append((_make_ad(rng, s), 2024, s))

    # 2026: 64 varied ads (~81% senior-associated)
    for _ in range(64):
        s = int(rng.random() < 0.81)
        rows.append((_make_ad(rng, s), 2026, s))

    return pd.DataFrame(rows, columns=["text", "year", "senior"])


if __name__ == "__main__":
    print("Running on a SYNTHETIC corpus (30 duplicate 2019 templates planted).")
    print("Replace `_synthetic_corpus()` with your real DataFrame to use.\n")
    demo = _synthetic_corpus()
    _ = analyze(demo, n_boot=2000, seed=1)
