"""Dataset loading and human-readable labels for the German Credit data."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"

TARGET = "class"
POSITIVE = "good"  # 1 = approve, 0 = reject

# Protected attributes are excluded from the model so the study does not show
# participants explanations that rest on sex, marital status or nationality.
EXCLUDED = ["personal_status", "foreign_worker"]

NUMERIC = [
    "duration", "credit_amount", "installment_commitment", "residence_since",
    "age", "existing_credits", "num_dependents",
]

LABELS = {
    "checking_status": "Checking account balance",
    "duration": "Loan duration",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Loan amount",
    "savings_status": "Savings balance",
    "employment": "Time in current job",
    "installment_commitment": "Repayment burden (1-4)",
    "other_parties": "Co-applicant or guarantor",
    "residence_since": "Years at current address",
    "property_magnitude": "Most valuable property",
    "age": "Age",
    "other_payment_plans": "Other instalment plans",
    "housing": "Housing",
    "existing_credits": "Existing loans at this bank",
    "job": "Job type",
    "num_dependents": "Number of dependents",
    "own_telephone": "Registered telephone",
}

VALUE_TEXT = {
    "checking_status": {
        "<0": "overdrawn (below 0 DM)", "0<=X<200": "0-200 DM",
        ">=200": "200 DM or more", "no checking": "no checking account",
    },
    "savings_status": {
        "<100": "under 100 DM", "100<=X<500": "100-500 DM",
        "500<=X<1000": "500-1,000 DM", ">=1000": "1,000 DM or more",
        "no known savings": "no known savings",
    },
    "employment": {
        "unemployed": "unemployed", "<1": "less than 1 year",
        "1<=X<4": "1-4 years", "4<=X<7": "4-7 years", ">=7": "7 years or more",
    },
    "own_telephone": {"yes": "yes", "none": "no"},
    "other_parties": {"none": "none"},
    "other_payment_plans": {"none": "none"},
}

UNITS = {
    "duration": "{:.0f} months",
    "credit_amount": "{:,.0f} DM",
    "age": "{:.0f} years",
    "residence_since": "{:.0f} years",
}


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return (features, target) with target 1 = good credit (approve)."""
    df = fetch_openml("credit-g", version=1, as_frame=True, data_home=DATA_DIR / "openml").frame
    y = (df[TARGET] == POSITIVE).astype(int)
    X = df.drop(columns=[TARGET, *EXCLUDED])
    for col in X.columns:
        if col in NUMERIC:
            X[col] = X[col].astype(float)
        else:
            X[col] = X[col].astype(str)
    return X, y


def categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in NUMERIC]


def label(feature: str) -> str:
    return LABELS.get(feature, feature.replace("_", " ").capitalize())


def value_text(feature: str, value) -> str:
    if feature in UNITS:
        return UNITS[feature].format(float(value))
    if feature in NUMERIC:
        return f"{float(value):.0f}"
    return VALUE_TEXT.get(feature, {}).get(str(value), str(value).replace("/", " / "))
