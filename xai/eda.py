"""Exploratory data analysis of the German Credit dataset.

Writes artifacts/eda/summary.json and three figures: class balance, numeric
distributions by outcome, and good-credit rate by checking account status.

Run: uv run python -m xai.eda
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import ARTIFACTS, NUMERIC, categorical_columns, label, load_dataset, value_text

OUT = ARTIFACTS / "eda"
GOOD, BAD = "#1F5FA8", "#C4661F"


def main():
    X, y = load_dataset()
    OUT.mkdir(parents=True, exist_ok=True)
    cats = categorical_columns(X)

    summary = {
        "rows": len(X),
        "features_used": X.shape[1],
        "numeric_features": NUMERIC,
        "categorical_features": cats,
        "class_balance": {"good": int(y.sum()), "bad": int((1 - y).sum()), "good_share": round(float(y.mean()), 3)},
        "missing_values": int(X.isna().sum().sum()),
        "numeric": X[NUMERIC].describe().round(2).to_dict(),
        "good_rate_by_category": {
            c: {value_text(c, v): round(float(r), 3) for v, r in y.groupby(X[c]).mean().items()} for c in cats
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # Class balance
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Good credit", "Bad credit"], [summary["class_balance"]["good"], summary["class_balance"]["bad"]],
           color=[GOOD, BAD])
    ax.set_ylabel("Applicants")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "class_balance.png", dpi=150)
    plt.close(fig)

    # Numeric distributions by outcome
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
    for ax, col in zip(axes.flat, NUMERIC):
        ax.hist([X.loc[y == 1, col], X.loc[y == 0, col]], bins=15, color=[GOOD, BAD],
                label=["Good", "Bad"], stacked=True)
        ax.set_title(label(col), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "numeric_distributions.png", dpi=150)
    plt.close(fig)

    # Good-credit rate by checking account status (the strongest single factor in the model)
    rates = summary["good_rate_by_category"]["checking_status"]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(list(rates), [r * 100 for r in rates.values()], color=GOOD)
    ax.axvline(summary["class_balance"]["good_share"] * 100, color="#1E2A33", linewidth=1, linestyle="--")
    ax.set_xlabel("Applicants with good credit (%)")
    ax.set_title(label("checking_status"), fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "good_rate_by_checking_status.png", dpi=150)
    plt.close(fig)

    print(f"{summary['rows']} applicants, {summary['features_used']} features used, "
          f"{summary['missing_values']} missing values, {summary['class_balance']['good_share']:.0%} good credit")
    print(f"Saved summary and figures to {OUT}")


if __name__ == "__main__":
    main()
