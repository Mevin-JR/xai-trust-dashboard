"""Select study cases and precompute LIME, SHAP and plain-language explanations.

Explanations are computed once and saved, so every participant sees exactly the
same explanation and the dashboard responds instantly.

Run: uv run python -m xai.explain   (after xai.train)
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from . import narrative
from .data import ARTIFACTS, NUMERIC, label, value_text

SEED = 42
LIME_SAMPLES = 5000
LIME_FEATURES = 8


def shap_by_feature(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP values summed back from one-hot columns to the original features."""
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    Xt = prep.transform(X)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Xt)
    base = explainer.expected_value
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    if values.ndim == 3:  # (rows, columns, classes)
        values = values[:, :, 1]
    if np.ndim(base) > 0:
        base = np.ravel(base)[-1]

    groups, start = [], 0
    for name, _, cols in prep.transformers_:
        if name == "num":
            for _ in cols:
                groups.append((start, start + 1))
                start += 1
        elif name == "cat":
            for cats in prep.named_transformers_["cat"].categories_:
                groups.append((start, start + len(cats)))
                start += len(cats)
    order = [c for name, _, cols in prep.transformers_ if name in ("num", "cat") for c in cols]
    summed = np.column_stack([values[:, a:b].sum(axis=1) for a, b in groups])
    # Reorder to match X's column order.
    summed = summed[:, [order.index(c) for c in X.columns]]
    return summed, float(base)


class LimeAdapter:
    """LIME needs a numeric matrix; categorical columns are encoded as integer codes."""

    def __init__(self, pipeline, X_train: pd.DataFrame):
        self.pipeline = pipeline
        self.columns = list(X_train.columns)
        self.cats = {c: sorted(X_train[c].unique()) for c in self.columns if c not in NUMERIC}
        cat_idx = [self.columns.index(c) for c in self.cats]
        self.explainer = LimeTabularExplainer(
            self.encode(X_train),
            feature_names=[label(c) for c in self.columns],
            categorical_features=cat_idx,
            categorical_names={self.columns.index(c): [value_text(c, v) for v in v_list]
                               for c, v_list in self.cats.items()},
            class_names=["Reject", "Approve"],
            discretize_continuous=True,
            random_state=SEED,
        )

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for c, values in self.cats.items():
            X[c] = X[c].map({v: i for i, v in enumerate(values)})
        return X[self.columns].to_numpy(dtype=float)

    def decode(self, arr: np.ndarray) -> pd.DataFrame:
        X = pd.DataFrame(arr, columns=self.columns)
        for c, values in self.cats.items():
            X[c] = [values[int(round(i))] for i in X[c]]
        return X

    def predict_proba(self, arr: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(self.decode(arr))

    def explain(self, row: pd.DataFrame) -> list[dict]:
        exp = self.explainer.explain_instance(
            self.encode(row)[0], self.predict_proba, num_features=LIME_FEATURES,
            num_samples=LIME_SAMPLES, labels=(1,))
        return [{"feature": self.columns[i], "rule": rule, "weight": float(w)}
                for (i, w), (rule, _) in zip(exp.as_map()[1], exp.as_list(label=1))]


def pick_cases(y_true: pd.Series, proba: np.ndarray) -> dict[str, list]:
    """Group test rows by outcome: approve/reject x correct/wrong."""
    rng = np.random.default_rng(SEED)
    pred = proba >= 0.5
    groups = {
        "approve_correct": (pred & (y_true == 1)), "reject_correct": (~pred & (y_true == 0)),
        "approve_wrong": (pred & (y_true == 0)), "reject_wrong": (~pred & (y_true == 1)),
    }
    sizes = {"approve_correct": 3, "reject_correct": 3, "approve_wrong": 2, "reject_wrong": 2}
    # Avoid trivially extreme cases so the explanations have something to show.
    moderate = (proba > 0.15) & (proba < 0.92)
    return {g: list(rng.choice(y_true.index[mask & moderate], sizes[g], replace=False))
            for g, mask in groups.items()}


def save_shap_plots(pipeline, X_test: pd.DataFrame, cases: dict, base: float, probability: bool) -> dict[str, float]:
    """Global summary (mean |SHAP| over the test set) and a waterfall plot per study case."""
    values, _ = shap_by_feature(pipeline, X_test)
    importance = pd.Series(np.abs(values).mean(axis=0), index=X_test.columns).sort_values()

    fig, ax = plt.subplots(figsize=(7, 6))
    scale, unit = (100, "percentage points of approval chance") if probability else (1, "log-odds of approval")
    ax.barh([label(c) for c in importance.index], importance.values * scale, color="#1F5FA8")
    ax.set_xlabel(f"Mean |SHAP value| ({unit})")
    ax.set_title(f"Global feature importance, test set ({len(X_test)} applicants)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "shap_global.png", dpi=150)
    plt.close(fig)

    folder = ARTIFACTS / "waterfall"
    folder.mkdir(exist_ok=True)
    for cid, case in cases.items():
        feats = list(case["shap"])
        exp = shap.Explanation(values=np.array([case["shap"][f] for f in feats]), base_values=base,
                               feature_names=[f"{label(f)} = {case['applicant'][f]}" for f in feats])
        shap.plots.waterfall(exp, max_display=10, show=False)
        plt.gcf().savefig(folder / f"{cid}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    return {c: round(float(v), 5) for c, v in importance.sort_values(ascending=False).items()}


def main():
    bundle = joblib.load(ARTIFACTS / "model.joblib")
    pipe, X_train, X_test, y_test = bundle["pipeline"], bundle["X_train"], bundle["X_test"], bundle["y_test"]
    proba = pipe.predict_proba(X_test)[:, 1]
    groups = pick_cases(y_test, proba)

    # Four study blocks: one case to study with an explanation, one new case
    # whose decision the participant predicts (forward simulation).
    g = {k: [str(i) for i in v] for k, v in groups.items()}
    blocks = [
        {"explained": g["approve_correct"][0], "forward": g["reject_correct"][0]},
        {"explained": g["reject_correct"][1], "forward": g["approve_correct"][1]},
        {"explained": g["approve_wrong"][0], "forward": g["reject_correct"][2]},
        {"explained": g["reject_wrong"][0], "forward": g["approve_correct"][2]},
    ]
    practice = [g["approve_wrong"][1], g["reject_wrong"][1]]

    ids = [int(i) for v in groups.values() for i in v]
    rows = X_test.loc[ids]
    shap_vals, base = shap_by_feature(pipe, rows)
    lime = LimeAdapter(pipe, X_train)
    threshold = 0.0 if bundle["name"] == "xgboost" else 0.5  # log-odds vs probability output

    cases = {}
    for n, idx in enumerate(ids):
        row = rows.loc[[idx]]
        p = float(pipe.predict_proba(row)[0, 1])
        approved = p >= 0.5
        factors = [narrative.Factor(c, label(c), value_text(c, row.iloc[0][c]), float(shap_vals[n, j]))
                   for j, c in enumerate(rows.columns)]
        cases[str(idx)] = {
            "id": str(idx),
            "applicant": {c: value_text(c, row.iloc[0][c]) for c in rows.columns},
            "proba_approve": p,
            "approved": bool(approved),
            "actual_good": bool(y_test.loc[idx] == 1),
            "shap": {c: float(shap_vals[n, j]) for j, c in enumerate(rows.columns)},
            "lime": lime.explain(row),
            "narrative": narrative.explain(factors, approved, max(p, 1 - p), baseline_approves=base >= threshold),
        }
        print(f"case {idx}: {'approve' if approved else 'reject'} p={p:.2f} actual_good={cases[str(idx)]['actual_good']}")

    global_importance = save_shap_plots(pipe, X_test, cases, base, probability=bundle["name"] != "xgboost")

    out = {
        "model": bundle["name"],
        "shap_global": global_importance,
        "shap_base": base,
        "shap_units": "log-odds of approval" if bundle["name"] == "xgboost" else "probability of approval",
        "labels": {c: label(c) for c in rows.columns},
        "blocks": blocks,
        "practice": practice,
        "cases": cases,
    }
    (ARTIFACTS / "explanations.json").write_text(json.dumps(out, indent=2))
    print(f"Saved {len(cases)} cases to {ARTIFACTS / 'explanations.json'}")


if __name__ == "__main__":
    main()
