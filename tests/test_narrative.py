from xai.narrative import Factor, explain, strength


def factors():
    return [
        Factor("duration", "Loan duration", "48 months", -0.20),
        Factor("checking_status", "Checking account balance", "no checking account", 0.12),
        Factor("age", "Age", "35 years", 0.03),
        Factor("purpose", "Loan purpose", "new car", -0.05),
    ]


def test_reasons_follow_shap_order_and_sign():
    out = explain(factors(), approved=False, confidence=0.6)
    assert [r.split("**")[1] for r in out["reasons"]] == ["Loan duration", "Checking account balance", "Loan purpose"]
    assert "towards rejection" in out["reasons"][0]
    assert "towards approval" in out["reasons"][1]
    assert "towards rejection" in out["reasons"][2]


def test_strength_wording():
    assert strength(0.5) == "strongly"
    assert strength(0.15) == "moderately"
    assert strength(0.05) == "slightly"


def test_headline_and_summary_match_decision():
    out = explain(factors(), approved=False, confidence=0.6, baseline_approves=False)
    assert "**reject**" in out["headline"] and "60%" in out["headline"]
    assert "towards rejection outweighed" in out["summary"]


def test_baseline_carries_decision():
    # Net push is towards rejection, but the model's default leaning keeps it at approve.
    out = explain(factors(), approved=True, confidence=0.55, baseline_approves=True)
    assert "usual leaning towards approval" in out["summary"]


def test_no_causal_language():
    out = explain(factors(), approved=False, confidence=0.6)
    text = " ".join([out["headline"], *out["reasons"], out["summary"]]).lower()
    assert "because" not in text and "caused" not in text
