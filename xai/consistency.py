"""How consistent are LIME and SHAP on the study cases?

For each case, compares the features each method ranks highest (by absolute
weight): whether the top feature matches, the overlap of the top three, and
whether the shared features point in the same direction.

Run: uv run python -m xai.consistency   (after xai.explain)
"""
import json

import pandas as pd

from .data import ARTIFACTS, label

TOP_K = 3


def compare_case(case: dict, k: int = TOP_K) -> dict:
    shap_rank = sorted(case["shap"], key=lambda f: abs(case["shap"][f]), reverse=True)
    lime = {r["feature"]: r["weight"] for r in case["lime"]}
    lime_rank = sorted(lime, key=lambda f: abs(lime[f]), reverse=True)
    shared = set(shap_rank[:k]) & set(lime_rank[:k])
    same_sign = sum((case["shap"][f] > 0) == (lime[f] > 0) for f in shared)
    return {
        "case": case["id"],
        "shap_top": shap_rank[0],
        "lime_top": lime_rank[0],
        "top_match": shap_rank[0] == lime_rank[0],
        f"top{k}_overlap": len(shared),
        "shared_same_direction": f"{same_sign}/{len(shared)}",
        "shap_top3": ", ".join(label(f) for f in shap_rank[:k]),
        "lime_top3": ", ".join(label(f) for f in lime_rank[:k]),
    }


def main():
    exp = json.loads((ARTIFACTS / "explanations.json").read_text())
    table = pd.DataFrame([compare_case(c) for c in exp["cases"].values()])
    summary = {
        "cases": len(table),
        "top_feature_agreement": int(table["top_match"].sum()),
        f"mean_top{TOP_K}_overlap": round(float(table[f"top{TOP_K}_overlap"].mean()), 2),
        "disagreeing_cases": table.loc[~table["top_match"], "case"].tolist(),
    }
    table.to_csv(ARTIFACTS / "lime_shap_consistency.csv", index=False)
    (ARTIFACTS / "lime_shap_consistency.json").write_text(json.dumps(summary, indent=2))

    print(table[["case", "top_match", f"top{TOP_K}_overlap", "shared_same_direction"]].to_string(index=False))
    print(f"\nSame top feature in {summary['top_feature_agreement']} of {summary['cases']} cases; "
          f"mean top-{TOP_K} overlap {summary[f'mean_top{TOP_K}_overlap']} features; "
          f"disagreeing: {', '.join(summary['disagreeing_cases']) or 'none'}")


if __name__ == "__main__":
    main()
