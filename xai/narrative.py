"""Plain-language explanations generated from SHAP values.

Sentences come from fixed templates filled with the SHAP attributions, so every
statement is faithful to the model: the features, their order and their
direction always match the SHAP values. Wording describes what influenced the
model, never real-world causes.
"""
from dataclasses import dataclass

TOP_K = 3
STRONG, MODERATE = 0.25, 0.10  # share of total |SHAP| attribution


@dataclass
class Factor:
    feature: str
    label: str
    value: str
    shap: float


def strength(share: float) -> str:
    if share >= STRONG:
        return "strongly"
    if share >= MODERATE:
        return "moderately"
    return "slightly"


def top_factors(factors: list[Factor], k: int = TOP_K) -> list[Factor]:
    return sorted(factors, key=lambda f: abs(f.shap), reverse=True)[:k]


def explain(factors: list[Factor], approved: bool, confidence: float, baseline_approves: bool = True) -> dict:
    """Return the decision sentence, one sentence per top factor, and a summary.

    `baseline_approves` says whether the model leans towards approval for an
    average applicant (the SHAP base value), which decides how the balance of
    factors is described.
    """
    total = sum(abs(f.shap) for f in factors) or 1.0
    decision = "approve" if approved else "reject"
    headline = f"The system recommends to **{decision}** this application ({confidence:.0%} confidence)."

    reasons = []
    for f in top_factors(factors):
        direction = "towards approval" if f.shap > 0 else "towards rejection"
        reasons.append(f"**{f.label}** ({f.value}) {strength(abs(f.shap) / total)} pushed the decision {direction}.")

    pro = sum(f.shap for f in factors if f.shap > 0)
    con = -sum(f.shap for f in factors if f.shap < 0)
    if approved == baseline_approves and (pro >= con) != approved:
        # The default leaning carried the decision despite the net push the other way.
        lean = "approval" if approved else "rejection"
        summary = (f"Some factors pointed the other way, but not enough to change the system's "
                   f"usual leaning towards {lean} for a typical applicant.")
    elif approved:
        summary = "Overall, the factors pointing towards approval outweighed those pointing towards rejection."
    else:
        summary = "Overall, the factors pointing towards rejection outweighed those pointing towards approval."
    if min(pro, con) / max(pro, con, 1e-9) > 0.8:
        summary += " The two sides were closely balanced, so this was a borderline case."
    return {"headline": headline, "reasons": reasons, "summary": summary}
