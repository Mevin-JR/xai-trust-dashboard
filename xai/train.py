"""Train and compare candidate models; save the chosen black-box model.

Run: uv run python -m xai.train
"""
import json

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import ARTIFACTS, NUMERIC, categorical_columns, load_dataset

SEED = 42
SCORING = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}


def preprocessor(cat_cols, scale_numeric=False):
    num = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def candidates(cat_cols):
    return {
        "logistic_regression": (
            Pipeline([("prep", preprocessor(cat_cols, scale_numeric=True)),
                      ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline([("prep", preprocessor(cat_cols)),
                      ("model", RandomForestClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1))]),
            {"model__n_estimators": [300], "model__max_depth": [6, 10, None], "model__min_samples_leaf": [1, 3]},
        ),
        "xgboost": (
            Pipeline([("prep", preprocessor(cat_cols)),
                      ("model", XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1))]),
            {"model__n_estimators": [200, 400], "model__max_depth": [3, 4],
             "model__learning_rate": [0.05, 0.1], "model__scale_pos_weight": [1.0, 0.43]},
        ),
    }


def main():
    X, y = load_dataset()
    cat_cols = categorical_columns(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, (pipe, grid) in candidates(cat_cols).items():
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X_train, y_train)
        scores = cross_validate(search.best_estimator_, X_train, y_train, cv=cv, scoring=SCORING)
        best = search.best_estimator_
        proba = best.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "best_params": search.best_params_,
            "cv": {k: round(float(np.mean(scores[f"test_{k}"])), 3) for k in SCORING},
            "test": {
                "accuracy": round(accuracy_score(y_test, pred), 3),
                "f1": round(f1_score(y_test, pred), 3),
                "roc_auc": round(roc_auc_score(y_test, proba), 3),
            },
        }
        fitted[name] = best
        print(f"{name:20s} CV {results[name]['cv']}  test {results[name]['test']}")

    # The study's "black box" is the stronger of the two ensemble models;
    # logistic regression is reported as the interpretable baseline.
    chosen = max(["random_forest", "xgboost"], key=lambda n: results[n]["cv"]["roc_auc"])
    print(f"Chosen black-box model: {chosen}")

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"name": chosen, "pipeline": fitted[chosen], "X_train": X_train, "X_test": X_test,
                 "y_test": y_test}, ARTIFACTS / "model.joblib")
    (ARTIFACTS / "model_comparison.json").write_text(json.dumps({"chosen": chosen, "results": results}, indent=2))


if __name__ == "__main__":
    main()
