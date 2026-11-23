# XAI Trust Dashboard

Practicum prototype for *Evaluating the Effectiveness of Explainable AI in Improving User Trust and
Understanding of Machine Learning Predictions*. It is both the deliverable and the user-study platform:
participants see loan decisions from a black-box model with no explanation, a LIME explanation, a SHAP
explanation, or a plain-language explanation generated from SHAP values.

## Run

```bash
uv sync                            # install dependencies (macOS also needs: brew install libomp)
uv run python -m xai.eda           # dataset summary and figures -> artifacts/eda/
uv run python -m xai.train         # train and compare models -> artifacts/model.joblib, model_comparison.json
uv run python -m xai.explain       # pick study cases, precompute explanations, SHAP global + waterfall plots
uv run python -m xai.consistency   # LIME vs SHAP agreement on the study cases -> artifacts/lime_shap_consistency.*
uv run streamlit run app.py        # open http://localhost:8501
uv run python -m xai.analysis      # statistical analysis of completed responses -> artifacts/analysis/
uv run pytest                      # tests for the plain-language generator and the analysis
```

- `http://localhost:8501` opens **Explore** mode (demo, model metrics, CSV export of responses).
- `http://localhost:8501/?mode=study` is the participant link: study flow only, no sidebar.

## How it works

| File | Purpose |
| --- | --- |
| `xai/data.py` | Loads German Credit (OpenML `credit-g`), readable feature names and values |
| `xai/train.py` | Logistic Regression baseline vs Random Forest vs XGBoost (5-fold CV grid search); keeps the better ensemble as the black box |
| `xai/explain.py` | Selects 10 test cases (correct and incorrect approvals/rejections), computes SHAP (TreeExplainer, summed back to original features) and LIME (fixed seed, 5,000 samples) |
| `xai/narrative.py` | Plain-language layer: template sentences from the top 3 SHAP factors, strength wording from each factor's share of total attribution |
| `xai/eda.py` | Dataset summary (class balance, missing values, distributions, good-credit rate by category) |
| `xai/consistency.py` | Compares LIME and SHAP top factors per case (top-1 match, top-3 overlap, direction) |
| `xai/analysis.py` | Friedman / Cochran's Q across conditions, Wilcoxon vs black box (Holm), usefulness ranks, Spearman satisfaction vs trust |
| `xai/storage.py` | Anonymous SQLite storage, balanced Latin-square order of the four conditions |
| `app.py` | Streamlit dashboard: Explore and Study modes |

**Design choices**

- `personal_status` (sex / marital status) and `foreign_worker` are excluded from the model so no
  explanation rests on protected attributes.
- Explanations are precomputed so every participant sees identical explanations (LIME is otherwise
  random between runs) and the app responds instantly.
- Plain-language sentences only restate SHAP values (feature, direction, relative strength) and avoid
  causal wording, so they are always faithful to the model.

## Study flow (about 15-20 minutes)

1. Consent page, then two background questions (AI familiarity, lending experience).
2. Four parts, one per condition, in counterbalanced order. Each part:
   - **Review**: an applicant, the system's recommendation and the condition's explanation, then
     "which factor mattered most?" (scored against the top SHAP feature), 6 trust items adapted from
     Jian et al. (2000), a self-rated understanding item, and 4 explanation-satisfaction items adapted
     from Hoffman et al. (2018) for the explained conditions.
   - **Predict**: a new applicant without a recommendation; the participant predicts the system's
     decision (forward simulation, scored automatically).
3. Rank LIME, SHAP and plain language by usefulness; optional comment.

Responses are stored in long format in `data/responses.db` (`pid, block, condition, case_id, item, value`),
with scale answers as numbers 1-7. Reverse-scored trust items end in `_rev`.

## Current results (test set, 200 applicants)

See `artifacts/model_comparison.json`. Random Forest was chosen (CV ROC-AUC 0.796); Logistic
Regression scored close (0.788), which is worth discussing as part of the accuracy-interpretability trade-off.
