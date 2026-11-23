"""End-to-end checks of storage and the Streamlit app, each on a temporary SQLite database."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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
    for box, rank in zip(at.selectbox, ["1 - most useful", "2", "3 - least useful"]):
        box.set_value(rank)
    at.run()
    at.button[0].click().run()
    assert at.title[0].value == "Thank you!"

    df = storage.export()
    assert df["completed_at"].notna().all()
    assert set(df["condition"]) == {"background", "final", *storage.CONDITIONS}
    scores = analysis.score(df)
    assert len(scores) == 4 and scores["trust"].notna().all()


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
