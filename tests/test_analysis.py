import numpy as np
import pandas as pd

from xai import analysis, storage


def simulated_responses(n=20, seed=0, effect=2):
    """Responses in the same long format as storage.export(), with trust raised by `effect` when explained."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n):
        pid, order_idx = f"p{p}", p % 4
        order = [storage.CONDITIONS[i] for i in storage.ORDERS[order_idx]]
        for block, cond in enumerate(order):
            base = 3 + (effect if cond != "black_box" else 0)
            items = {k: int(np.clip(base + rng.integers(-1, 2), 1, 7)) for k in analysis.TRUST}
            for k in ("trust_5_rev", "trust_6_rev"):
                items[k] = 8 - items[k]
            items["understand_self"] = int(np.clip(base + rng.integers(-1, 2), 1, 7))
            items["top_factor_correct"] = bool(rng.random() < (0.8 if cond != "black_box" else 0.3))
            if cond != "black_box":
                items |= {k: int(np.clip(base + rng.integers(-1, 2), 1, 7)) for k in analysis.SATISFACTION}
            items |= {"forward_correct": bool(rng.random() < 0.6), "forward_guess": "Approve", "forward_confidence": 4}
            rows += [(pid, block, cond, "1", k, str(v)) for k, v in items.items()]
        ranks = rng.permutation([1, 2, 3])
        rows += [(pid, 4, "final", "", k, str(r)) for k, r in zip(analysis.RANKS, ranks)]
        rows += [(pid, 4, "final", "", "comment", "")]
    df = pd.DataFrame(rows, columns=["pid", "block", "condition", "case_id", "item", "value"])
    df["completed_at"] = "2026-11-16T10:00:00+00:00"
    return df


def test_detects_trust_effect():
    res = analysis.analyse(simulated_responses())
    assert res["participants"] == 20
    assert res["trust"]["friedman"]["p"] < 0.05
    assert all(v["p_holm"] < 0.05 for v in res["trust"]["vs_black_box"].values())
    means = res["condition_means"]
    assert means["lime"]["trust"] > means["black_box"]["trust"]


def test_reverse_items_are_recoded():
    # Simulation stores reverse items already flipped, so recoding must restore the same level.
    scores = analysis.score(simulated_responses(n=4, effect=0))
    assert scores["trust"].between(2, 4).all()


def test_no_effect_is_not_significant():
    res = analysis.analyse(simulated_responses(effect=0, seed=3))
    assert res["trust"]["friedman"]["p"] > 0.05


def test_too_few_participants():
    res = analysis.analyse(simulated_responses(n=2))
    assert "note" in res


def test_satisfaction_only_for_explained_conditions():
    scores = analysis.score(simulated_responses(n=4))
    assert scores.loc[scores["condition"] == "black_box", "satisfaction"].isna().all()
    assert scores.loc[scores["condition"] != "black_box", "satisfaction"].notna().all()
