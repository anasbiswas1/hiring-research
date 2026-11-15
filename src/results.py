"""Table II and Fig. 2 of the paper from the final ratings, with template-corrected intervals.

Primary estimate per year: advertisements whose title carries a junior cue ("title cue" route),
the only definition that exists in every source. Secondary: advertisements selected only by the
platform's experience-level field ("platform level" route). When a stratum was sampled before
rating, the year-level figure reweights the strata by their full pool sizes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dedup_inference import _cluster_boot, cluster_templates, clustered_trend, duplication_report

SOURCE_LABEL = {2019: "Monster USA", 2024: "LinkedIn", 2026: "LinkedIn"}
ROUTES = ("title cue", "platform level")


def route_of(why) -> str:
    return "title cue" if str(why).startswith("title cue") else "platform level"


def _stratum(g: pd.DataFrame, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    senior = g.senior.to_numpy()
    cluster = g.cluster.to_numpy()
    reps = _cluster_boot(senior, cluster, n_boot, rng)
    naive = np.array([senior[rng.integers(0, len(senior), len(senior))].mean() for _ in range(n_boot)])
    deff = float(np.var(reps, ddof=1) / max(np.var(naive, ddof=1), 1e-12))
    return {"n_ads": int(len(g)), "n_templates": int(g.cluster.nunique()), "point": float(senior.mean()),
            "cluster_ci": (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))),
            "design_effect": deff, "effective_n": float(len(g) / deff) if deff > 0 else float("nan"),
            "_reps": reps}


def compute(ratings: pd.DataFrame, ads: pd.DataFrame, n_boot: int = 20000, seed: int = 7,
            threshold: float = 0.85, pool_sizes: pd.DataFrame | None = None) -> dict:
    """ratings: ad_id, year, why, final (0/1). ads: ad_id, title, desc (clustering on full text).

    pool_sizes: optional frame year, route, N (full pool before any sampling cap) for reweighting.
    """
    df = ratings.merge(ads[["ad_id", "title", "desc"]], on="ad_id", how="left")
    if df.final.isna().any():
        raise ValueError("every advertisement needs a final label before the results are computed")
    df["senior"] = df.final.astype(int)
    df["route"] = df.why.map(route_of)
    df["text"] = df.title.astype(str) + ". " + df.desc.astype(str)
    df["cluster"] = cluster_templates(df.text, threshold=threshold)
    dup = duplication_report(df, "cluster", "year")

    strata, years = {}, {}
    for (y, r), g in df.groupby(["year", "route"]):
        if len(g) >= 5:
            strata[(int(y), r)] = _stratum(g, n_boot, seed)
    for y in sorted(df.year.unique()):
        y = int(y)
        parts = {r: strata[(y, r)] for r in ROUTES if (y, r) in strata}
        if pool_sizes is not None:
            w = {r: float(pool_sizes.query("year == @y and route == @r").N.sum()) for r in parts}
        else:
            w = {r: float(parts[r]["n_ads"]) for r in parts}
        tot = sum(w.values())
        if parts and tot > 0:
            reps = sum(w[r] / tot * parts[r]["_reps"] for r in parts)
            point = sum(w[r] / tot * parts[r]["point"] for r in parts)
            years[y] = {"point": float(point), "cluster_ci": (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))),
                        "n_ads": int(sum(p["n_ads"] for p in parts.values())), "weights": {r: w[r] / tot for r in parts},
                        "n_templates": int(df[df.year == y].cluster.nunique())}
    title_only = df[df.route == "title cue"]
    trend_title = clustered_trend(title_only, "senior", "year", "cluster") if title_only.year.nunique() >= 2 else {"ok": False, "reason": "fewer than two years"}
    trend_all = clustered_trend(df, "senior", "year", "cluster")
    return {"strata": strata, "years": years, "duplication": dup, "trend_title_cue": trend_title,
            "trend_all_routes_unweighted": trend_all, "frame": df}


def table2(res: dict) -> pd.DataFrame:
    rows = []
    for (y, r), s in sorted(res["strata"].items()):
        lo, hi = s["cluster_ci"]
        rows.append({"Year (source)": f"{y} ({SOURCE_LABEL.get(y, '')})", "Selection": r,
                     "n rated (templates)": f"{s['n_ads']} ({s['n_templates']})", "Share": f"{100 * s['point']:.1f}%",
                     "Template-corrected 95% CI": f"{100 * lo:.1f} to {100 * hi:.1f}%", "Effective n": f"{s['effective_n']:.0f}"})
    for y, s in sorted(res["years"].items()):
        if len(s["weights"]) > 1:
            lo, hi = s["cluster_ci"]
            rows.append({"Year (source)": f"{y} ({SOURCE_LABEL.get(y, '')})", "Selection": "all routes, weighted by pool size",
                         "n rated (templates)": f"{s['n_ads']} ({s['n_templates']})", "Share": f"{100 * s['point']:.1f}%",
                         "Template-corrected 95% CI": f"{100 * lo:.1f} to {100 * hi:.1f}%", "Effective n": ""})
    return pd.DataFrame(rows)


def _serif_font() -> str:
    """First installed Times-like family, so the figure matches the paper's body font."""
    from matplotlib import font_manager
    for name in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            return name
        except Exception:
            continue
    return "serif"


def fig2(res: dict, path: Path, dpi: int = 600) -> None:
    """Share by year: junior-titled advertisements (filled) and platform-level selections (hollow)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = _serif_font()
    strata = res["strata"]
    years = sorted({y for y, _ in strata})
    xs = {2019: 0.0, 2024: 1.5, 2026: 2.6}
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    lows = [100 * s["cluster_ci"][0] for s in strata.values()]
    ymin = max(0, 10 * int((min(lows) - 8) // 10))
    ax.axvspan(0.85, 3.15, color="#f0eee8", lw=0, zorder=0)
    ax.text(3.08, ymin + 1.2, "generative-AI era (ChatGPT released Nov 2022)", ha="right", va="bottom", fontsize=6.3, color="#555555")
    if ymin < 50:
        ax.axhline(50, color="#bbbbbb", lw=0.6, ls=":", zorder=0)
    style = {"title cue": dict(fmt="o", color="#2f5fc4", mfc="#2f5fc4", dx=-0.08, label="junior word in title"),
             "platform level": dict(fmt="s", color="#7a7a7a", mfc="white", dx=0.12, label="platform experience level only")}
    seen = set()
    for (y, r), s in sorted(strata.items()):
        st = style[r]
        x = xs.get(y, len(seen)) + st["dx"]
        p, (lo, hi) = 100 * s["point"], (100 * s["cluster_ci"][0], 100 * s["cluster_ci"][1])
        ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt=st["fmt"], color=st["color"], mfc=st["mfc"], ms=4.5,
                    capsize=3, lw=1.2, capthick=1.0, label=st["label"] if r not in seen else None, zorder=3)
        if r == "title cue":
            ax.text(x - 0.1, p, f"{p:.0f}%", va="center", ha="right", fontsize=7)
        seen.add(r)
    labels = []
    for y in years:
        s = strata.get((y, "title cue")) or strata.get((y, "platform level"))
        n_t = strata[(y, "title cue")]["n_ads"] if (y, "title cue") in strata else 0
        eff = f"\neffective ≈ {s['effective_n']:.0f}" if s["design_effect"] > 1.15 else ""
        labels.append((xs.get(y, 0), f"{y}\n{SOURCE_LABEL.get(y, '')}\nn = {n_t}{eff}"))
    ax.set_xticks([x for x, _ in labels])
    ax.set_xticklabels([t for _, t in labels], fontsize=7)
    ax.set_xlim(-0.45, 3.15)
    ax.set_ylim(ymin, 100)
    ax.set_yticks(range(ymin, 101, 10))
    ax.set_ylabel("Junior software advertisements asking for\nat least one senior responsibility (%)", fontsize=7.6)
    ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.5, length=2.5)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def summary(res: dict) -> dict:
    """JSON-ready numbers for the paper text."""
    out = {"strata": {}, "years": {}, "duplication": res["duplication"].to_dict(orient="records"),
           "trend_title_cue": {k: v for k, v in res["trend_title_cue"].items() if k != "note"},
           "trend_all_routes_unweighted": {k: v for k, v in res["trend_all_routes_unweighted"].items() if k != "note"}}
    for (y, r), s in res["strata"].items():
        out["strata"][f"{y}|{r}"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items() if not k.startswith("_")}
    for y, s in res["years"].items():
        out["years"][str(y)] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()}
    return out
