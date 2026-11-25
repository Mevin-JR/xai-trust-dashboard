"""End-to-end checks of storage and the Streamlit app, each on a temporary SQLite database."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.errors import AppTestError

from xai import analysis, storage

APP = str(Path(__file__).resolve().parent.parent / "app.py")
LABELS = json.loads((Path(APP).parent / "artifacts" / "explanations.json").read_text())["labels"]
FEATURE = {v: k for k, v in LABELS.items()}


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")


def test_storage_round_trip_and_latin_square_orders():
    orders = [storage.new_participant()[1] for _ in range(4)]
    assert all(sorted(o) == sorted(storage.CONDITIONS) for o in orders)
    assert [o[0] for o in orders] == storage.CONDITIONS  # each condition first exactly once
    pid = storage.new_participant()[0]
    storage.save(pid, 0, "lime", "732", {"trust_1": 5, "forward_correct": True})
    storage.complete(pid)
    df = storage.export()
    assert set(df["item"]) == {"trust_1", "forward_correct"}
    assert df["completed_at"].notna().all()
    assert storage.counts() == (5, 1)


def button(at, label):
    return next(b for b in at.button if b.label == label)


def pick(radio, i):
    """Radio options are display labels; the factor question stores feature names."""
    radio.set_value(FEATURE.get(radio.options[i], radio.options[i]))


def test_full_participant_session():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["mode"] = "study"
    at.run()
    at.checkbox[0].check().run()
    at.button[0].click().run()
    for r in at.radio:
        pick(r, 1)
    at.run()
    at.button[0].click().run()
    for _ in range(4):
        for r in at.radio:
            pick(r, 1)
        at.run()
        at.button[0].click().run()
        for r in at.radio:
            pick(r, 0)
        at.run()
        at.button[0].click().run()
        assert not at.exception
    ranks = at.radio
    for i, r in enumerate(ranks):  # a different rank for each explanation type
        r.set_value(r.options[i])
    at.run()
    button(at, "Submit my answers").click().run()
    assert at.title[0].value == "Thank you!"

    df = storage.export()
    assert df["completed_at"].notna().all()
    assert set(df["condition"]) == {"background", "final", *storage.CONDITIONS}
    scores = analysis.score(df)
    assert len(scores) == 4 and scores["trust"].notna().all()
    assert df.loc[df["item"] == "trust_1", "value"].astype(int).eq(2).all()  # "Disagree" stored as 2
    assert sorted(df.loc[df["item"].str.startswith("rank_"), "value"].astype(int)) == [1, 2, 3]


def test_researcher_view_needs_password():
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets["RESEARCHER_PASSWORD"] = "correct horse"
    at.run()
    assert at.title[0].value == "Researcher sign-in"
    assert not at.sidebar.radio  # no case register or responses before sign-in
    at.text_input[0].input("wrong")
    at.button[0].click().run()  # a form sends its fields only on submit
    assert at.error and at.title[0].value == "Researcher sign-in"
    at.text_input[0].input("correct horse")
    at.button[0].click().run()  # a form sends its fields only on submit
    assert at.title[0].value == "Loan application review"


def test_study_link_skips_password():
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets["RESEARCHER_PASSWORD"] = "correct horse"
    at.query_params["mode"] = "study"
    at.run()
    assert at.title[0].value == "How do people judge AI loan decisions?"


def test_cannot_start_without_consent():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["mode"] = "study"
    at.run()
    assert at.button[0].disabled
    with pytest.raises(AppTestError):  # like a browser, the harness cannot press a disabled button
        at.button[0].click()
    assert storage.counts() == (0, 0)
    at.checkbox[0].check().run()
    assert not at.button[0].disabled



def test_next_explains_what_is_missing():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["mode"] = "study"
    at.run()
    at.checkbox[0].check().run()
    at.button[0].click().run()
    assert at.button[0].disabled
    assert at.button[0].help == "Answer all questions to continue (2 left)."
    at.radio[0].set_value(at.radio[0].options[0]).run()
    assert at.button[0].help == "Answer all questions to continue (1 left)."
    at.radio[1].set_value(at.radio[1].options[0]).run()
    assert not at.button[0].disabled


def final_page():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["mode"] = "study"
    at.session_state["step"] = "final"
    at.session_state["pid"] = "test"
    return at.run()


def test_taken_rank_is_locked_on_other_explanations():
    at = final_page()
    at.radio[0].set_value("Most useful").run()
    css = " ".join(m.value for m in at.markdown if "<style>" in m.value and "Given to" in m.value)
    assert 'rank_shap_' in css and 'Given to LIME' in css and "pointer-events:none" in css


def test_choosing_a_taken_rank_clears_it_elsewhere():
    at = final_page()
    at.radio[0].set_value("Most useful").run()
    at.radio[1].set_value("Most useful").run()  # e.g. reached with the keyboard
    assert at.radio[0].value is None and at.radio[1].value == "Most useful"
    assert button(at, "Submit my answers").disabled


def test_rank_can_be_cleared_and_reused():
    at = final_page()
    at.radio[0].set_value("Most useful").run()
    assert [b.label for b in at.button].count("Clear") == 1  # only rows with an answer offer Clear
    button(at, "Clear").click().run()
    assert at.radio[0].value is None
    assert not any("Given to LIME" in m.value for m in at.markdown)  # the rank is free again
    at.radio[1].set_value("Most useful").run()
    assert at.radio[1].value == "Most useful" and at.radio[0].value is None



def test_info_button_on_every_study_page():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["mode"] = "study"
    at.run()
    assert any(b.label == "?" for b in at.button)
    at.checkbox[0].check().run()
    at.button[0].click().run()
    assert any(b.label == "?" for b in at.button)
