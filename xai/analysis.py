"""Statistical analysis of the study responses (the plan from the methodology chapter).

Per participant and condition it scores:
  trust          mean of the 6 trust items, reverse items recoded (6 - x)       1-5
  understand     self-rated understanding                                       1-5
  satisfaction   mean of the 4 explanation-satisfaction items (explained only)  1-5
  factor_correct picked the top SHAP feature as most influential                0/1
  forward_correct predicted the system's decision for a new applicant           0/1

Tests (only completed participants are included):
  RQ / SQ1-3  Friedman test across the four conditions for trust and understanding,
              Cochran's Q for the two accuracy measures, and Wilcoxon signed-rank
              tests of each explanation against the black box (Holm-corrected).
  SQ4         Friedman test on the usefulness ranks of LIME, SHAP and plain language.
  SQ5         Spearman correlation between explanation satisfaction and trust.
  Reliability Cronbach's alpha for the trust and satisfaction scales.

Run: uv run python -m xai.analysis
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from . import storage
from .data import ARTIFACTS

OUT = ARTIFACTS / "analysis"
CONDITIONS = storage.CONDITIONS
EXPLAINED = [c for c in CONDITIONS if c != "black_box"]
AGREE_POINTS = 5  # Strongly disagree (1) ... Strongly agree (5)
TRUST = ["trust_1", "trust_2", "trust_3", "trust_4", "trust_5_rev", "trust_6_rev"]
SATISFACTION = ["sat_1", "sat_2", "sat_3", "sat_4"]
RANKS = {"rank_lime": "lime", "rank_shap": "shap", "rank_plain-language": "plain_language"}
MIN_PARTICIPANTS = 3


def as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"True": 1, "False": 0}), errors="coerce")


def score(responses: pd.DataFrame) -> pd.DataFrame:
    """One row per participant x condition with the five outcome measures."""
    df = responses[responses["condition"].isin(CONDITIONS)]
    wide = df.pivot_table(index=["pid", "condition"], columns="item", values="value", aggfunc="first")
    wide = wide.apply(as_number)
    for item in TRUST:
        if item.endswith("_rev"):
            wide[item] = AGREE_POINTS + 1 - wide[item]
    out = pd.DataFrame(index=wide.index)
    out["trust"] = wide[TRUST].mean(axis=1)
    out["understand"] = wide["understand_self"]
    out["satisfaction"] = wide.reindex(columns=SATISFACTION).mean(axis=1)
    out["factor_correct"] = wide["top_factor_correct"]
    out["forward_correct"] = wide["forward_correct"]
    return out.reset_index()


def cronbach_alpha(items: pd.DataFrame) -> float | None:
    """Internal consistency of a scale; rows = responses, columns = items."""
    items = items.dropna()
    k = items.shape[1]
    if len(items) < 3 or k < 2 or items.sum(axis=1).var(ddof=1) == 0:
        return None
    return round(float(k / (k - 1) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1))), 3)


def item_responses(responses: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    """One row per participant x condition with the raw items (reverse items recoded)."""
    df = responses[responses["item"].isin(items)]
    wide = df.pivot_table(index=["pid", "condition"], columns="item", values="value", aggfunc="first").apply(as_number)
    for item in wide.columns:
        if item.endswith("_rev"):
            wide[item] = AGREE_POINTS + 1 - wide[item]
    return wide.reindex(columns=items)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def cochran_q(binary: pd.DataFrame) -> dict:
    """Cochran's Q for k related binary samples (rows = participants)."""
    x = binary.to_numpy(dtype=float)
    k = x.shape[1]
    col, row, total = x.sum(axis=0), x.sum(axis=1), x.sum()
    denom = k * total - (row ** 2).sum()
    if denom == 0:
        return {"Q": None, "p": None, "note": "no variation"}
    q = (k - 1) * (k * (col ** 2).sum() - total ** 2) / denom
    return {"Q": round(float(q), 3), "p": round(float(stats.chi2.sf(q, k - 1)), 4)}


def friedman(wide: pd.DataFrame) -> dict:
    if wide.nunique().le(1).all():
        return {"chi2": None, "p": None, "note": "no variation"}
    chi2, p = stats.friedmanchisquare(*[wide[c] for c in wide.columns])
    return {"chi2": round(float(chi2), 3), "p": round(float(p), 4)}


def versus_black_box(wide: pd.DataFrame) -> dict:
    raw = {}
    for c in EXPLAINED:
        diff = wide[c] - wide["black_box"]
        raw[c] = 1.0 if (diff == 0).all() else float(stats.wilcoxon(wide[c], wide["black_box"]).pvalue)
    adjusted = holm(raw)
    return {c: {"median_difference": round(float((wide[c] - wide["black_box"]).median()), 3),
                "p_holm": round(adjusted[c], 4)} for c in EXPLAINED}


def analyse(responses: pd.DataFrame) -> dict:
    completed = responses.dropna(subset=["completed_at"])
    n = completed["pid"].nunique()
    if n < MIN_PARTICIPANTS:
        return {"participants": n, "note": f"At least {MIN_PARTICIPANTS} completed participants are needed."}

    scores = score(completed)
    by_cond = lambda m: scores.pivot(index="pid", columns="condition", values=m)[CONDITIONS].dropna()

    means = scores.groupby("condition")[["trust", "understand", "satisfaction", "factor_correct",
                                         "forward_correct"]].mean().round(3).reindex(CONDITIONS)
    results = {"participants": n,
               "reliability_cronbach_alpha": {
                   "trust": cronbach_alpha(item_responses(completed, TRUST)),
                   "satisfaction": cronbach_alpha(item_responses(completed, SATISFACTION))},
               "condition_means": means.astype(object).where(means.notna(), None).to_dict(orient="index")}
    for m in ("trust", "understand"):
        wide = by_cond(m)
        results[m] = {"friedman": friedman(wide), "vs_black_box": versus_black_box(wide)}
    for m in ("factor_correct", "forward_correct"):
        results[m] = {"cochran_q": cochran_q(by_cond(m))}

    final = completed[completed["condition"] == "final"]
    ranks = final[final["item"].isin(RANKS)].pivot(index="pid", columns="item", values="value").apply(as_number)
    ranks = ranks.rename(columns=RANKS).dropna()
    results["usefulness_rank"] = {
        "mean_rank": ranks.mean().round(2).to_dict(),
        "ranked_first": (ranks == 1).sum().astype(int).to_dict(),
        "friedman": friedman(ranks) if len(ranks) >= MIN_PARTICIPANTS else None,
    }

    explained = scores[scores["condition"].isin(EXPLAINED)].dropna(subset=["satisfaction", "trust"])
    rho, p = stats.spearmanr(explained["satisfaction"], explained["trust"])
    results["satisfaction_vs_trust"] = {"spearman_rho": None if np.isnan(rho) else round(float(rho), 3),
                                        "p": None if np.isnan(p) else round(float(p), 4),
                                        "observations": len(explained)}
    results["_scores"] = scores
    return results


def main():
    results = analyse(storage.export())
    OUT.mkdir(parents=True, exist_ok=True)
    scores = results.pop("_scores", None)
    if scores is not None:
        scores.to_csv(OUT / "scores_by_condition.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
