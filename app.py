"""XAI Trust Dashboard: explore explanations, or run the user study.

Run: uv run streamlit run app.py
"""
import hmac
import html
import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from xai import storage, study_text
from xai.data import ARTIFACTS

st.set_page_config(page_title="Loan application review", layout="wide")

INK, MUTED, RULE = "#1E2A33", "#5C6B73", "#C9D3D6"
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;800&display=swap');
:root { --paper:#F5F7F4; --sheet:#FFFFFF; --ink:#1E2A33; --muted:#5C6B73; --rule:#C9D3D6;
        --approve:#1F5FA8; --reject:#C4661F; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp button, .stApp input, .stApp textarea { font-family:'Public Sans', sans-serif; }
.stApp h1 { font-weight:800; font-size:2.1rem; letter-spacing:-0.015em; line-height:1.15; }
.stApp h5 { font-weight:600; font-size:1.05rem; }
.block-container { max-width:1180px; padding-top:2.5rem; }

/* The application form: a single sheet with fill-in lines. */
.sheet { background:var(--sheet); border:1px solid var(--rule); border-top:5px solid var(--ink);
         padding:1.1rem 1.4rem 1.3rem; }
.sheet-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.9rem; }
.sheet-head h3 { margin:0; padding:0; font-size:1.1rem; font-weight:600; color:var(--ink); }
.sheet-head span { font-size:0.85rem; color:var(--muted); font-variant-numeric:tabular-nums; }
dl.fields { margin:0; }
dl.fields div { display:grid; grid-template-columns:minmax(9rem, 45%) 1fr; gap:1rem; align-items:baseline;
                border-bottom:1px dotted var(--rule); padding:0.38rem 0; }
dl.fields dt { font-size:0.84rem; color:var(--muted); font-weight:400; }
dl.fields dd { margin:0; font-size:0.95rem; font-weight:500; color:var(--ink); }

/* The system's assessment: a second sheet beside the application, coloured by its decision. */
.sheet.assessment { margin-bottom:1.1rem; }
.sheet.assessment.approve { border-top-color:var(--approve); --decision:var(--approve); }
.sheet.assessment.reject { border-top-color:var(--reject); --decision:var(--reject); }
.stApp .sheet p.decision { margin:0 0 0.9rem; font-size:2.6rem; font-weight:800; line-height:1; letter-spacing:-0.02em;
            color:var(--decision); }
.meter .track { position:relative; height:10px; border-radius:5px;
                background:linear-gradient(90deg, rgba(196,102,31,0.22) 50%, rgba(31,95,168,0.22) 50%); }
.meter .line { position:absolute; top:-5px; bottom:-5px; width:2px; margin-left:-1px; background:var(--ink); }
.meter .dot { position:absolute; top:50%; width:18px; height:18px; margin:-9px 0 0 -9px; border-radius:50%;
              background:var(--decision); border:3px solid var(--sheet); box-shadow:0 0 0 1px var(--decision); }
.meter .ends { display:flex; justify-content:space-between; margin-top:0.45rem; font-size:0.8rem; color:var(--muted); }
.stApp .meter p { margin:0.35rem 0 0; font-size:0.85rem; color:var(--muted); }
.stApp .sheet p.outcome { margin:0.8rem 0 0; padding-top:0.6rem; border-top:1px dotted var(--rule); font-size:0.9rem; }
.outcome b.wrong { color:var(--reject); }

/* Sidebar: the case register. */
[data-testid="stSidebar"] h2 { font-size:1.35rem; font-weight:800; letter-spacing:-0.01em; margin-bottom:0; }
[data-testid="stSidebar"] h4 { font-size:0.95rem; font-weight:600; margin:1.3rem 0 0.2rem; padding:0;
                               color:#E9EEEB; border-top:1px solid #3A4A55; padding-top:1rem; }
[data-testid="stSidebar"] { min-width:320px; max-width:320px; }
.st-key-register [data-testid="stElementContainer"], .st-key-register [data-testid="stRadio"],
.st-key-register [data-testid="stRadioGroup"], .st-key-register [data-testid="stRadioGroup"] > div,
.st-key-register label[data-testid="stRadioOption"] { width:100% !important; }
.st-key-register [role="radiogroup"] { gap:0; }
.st-key-filter [data-testid="stButtonGroup"] button { padding:4px 10px; }
.st-key-register [role="radiogroup"] label { align-self:stretch; width:auto; margin:0; padding:0.5rem 0.6rem; border-bottom:1px solid #3A4A55;
                                      cursor:pointer; }
.st-key-register [role="radiogroup"] label > div > div:first-child { display:none; }
.st-key-register [role="radiogroup"] label:hover { background:#2A3944; }
.st-key-register [role="radiogroup"] label:has(input:checked) { background:#2A3944; box-shadow:inset 3px 0 0 #8DB4E2; }
.st-key-register [role="radiogroup"] label:has(input:focus-visible) { outline:2px solid #8DB4E2; outline-offset:-2px; }

/* Survey: the questionnaire is another sheet in the file. */
[class*="st-key-sheet_"] { background:var(--sheet); border:1px solid var(--rule); border-top:5px solid var(--ink);
                           padding:0.6rem 1.4rem 1.2rem; margin-top:0.6rem; }
.q-section { margin:1.2rem 0 0.2rem; }
.q-section h4 { margin:0; padding:0; font-size:1.08rem; font-weight:700; color:var(--ink); }
.q-section p { margin:0.15rem 0 0; font-size:0.88rem; color:var(--muted); }
[class*="st-key-q_"] { border-top:1px dotted var(--rule); padding:0.9rem 0 0.4rem; }
.q-head { display:flex; gap:0.9rem; align-items:baseline; }
.q-num { flex:0 0 2rem; font-weight:800; font-size:1.05rem; color:var(--muted); font-variant-numeric:tabular-nums; }
.q-text { margin:0; font-size:1rem; font-weight:500; color:var(--ink); line-height:1.4; }
[class*="st-key-q_"] [data-testid="stRadio"], [class*="st-key-q_"] [data-testid="stTextArea"] { margin-left:2.9rem; }
@media (max-width:640px) {
  [class*="st-key-q_"] [data-testid="stRadio"], [class*="st-key-q_"] [data-testid="stTextArea"] { margin-left:0; }
}

/* Answer options as buttons, all the same width. */
[class*="st-key-q_"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]),
[class*="st-key-q_"] [data-testid="stElementContainer"]:has([data-testid="stTextArea"]) { width:100% !important; }
[class*="st-key-q_first"] { border-top:none; }
/* Streamlit wraps each option in its own div, so the radio group itself is the grid. Five equal columns:
   every scale reads in one row and every button on the page has the same width. */
[class*="st-key-q_"] [data-testid="stRadioGroup"] { display:grid !important; width:100%;
                                                    grid-template-columns:repeat(5, minmax(0, 1fr)); gap:0.45rem; }
[class*="st-key-q_"] label[data-testid="stRadioOption"] {
  width:100%; min-height:3.6rem; margin:0; padding:0.4rem 0.45rem; box-sizing:border-box;
  display:flex; align-items:center; justify-content:center; text-align:center;
  background:#EEF2F0; border:1px solid var(--rule); border-radius:6px; cursor:pointer; }
[class*="st-key-q_"] label[data-testid="stRadioOption"] > div > div:first-child { display:none; }
[class*="st-key-q_"] label[data-testid="stRadioOption"] p { font-size:0.82rem; line-height:1.25; color:var(--ink);
                                                             hyphens:auto; }
[class*="st-key-q_"] label[data-testid="stRadioOption"]:hover { border-color:var(--ink); }
[class*="st-key-q_"] label[data-testid="stRadioOption"]:has(input:checked) { background:var(--ink); border-color:var(--ink); }
[class*="st-key-q_"] label[data-testid="stRadioOption"]:has(input:checked) p { color:#FFFFFF; font-weight:600; }
/* Agreement scales: a lighter tint and a lighter selected slate than the decision and prediction questions. */
[class*="st-key-q_agree"] label[data-testid="stRadioOption"] { background:#F6F8F7; border-color:#D5DDDF; }
[class*="st-key-q_agree"] label[data-testid="stRadioOption"]:hover { border-color:#4E6472; }
[class*="st-key-q_agree"] label[data-testid="stRadioOption"]:has(input:checked) { background:#4E6472;
                                                                                 border-color:#4E6472; }
[class*="st-key-q_"] label[data-testid="stRadioOption"]:has(input:focus-visible) { outline:2px solid var(--approve);
                                                                                   outline-offset:2px; }
@media (max-width:640px) {
  [class*="st-key-q_"] [data-testid="stRadioGroup"] { grid-template-columns:1fr; }
  [class*="st-key-q_"] label[data-testid="stRadioOption"] { min-height:2.8rem; }
}

/* "Clear" under a ranking row: a quiet text control, aligned with the options. */
[class*="st-key-clear_"] { margin-left:2.9rem; }
[class*="st-key-clear_"] button { min-height:0; padding:0.1rem 0; color:var(--muted); }
[class*="st-key-clear_"] button p { font-size:0.84rem; text-decoration:underline; text-underline-offset:2px; }
[class*="st-key-clear_"] button:hover p { color:var(--ink); }
@media (max-width:640px) { [class*="st-key-clear_"] { margin-left:0; } }

/* Why the button is disabled. */
.st-key-gate [data-testid="stElementContainer"] { align-self:flex-end; width:auto !important; }
.gate { margin:1rem 0 0.4rem; font-size:0.92rem; color:var(--reject); font-weight:500; text-align:right; }
.gate.ok { color:var(--muted); font-weight:400; }
.st-key-gate_left .gate { text-align:left; }

/* Participant information: the one floating element. */
.st-key-info_fab { position:fixed; right:1.5rem; bottom:1.5rem; z-index:1000; width:auto !important; }
.st-key-info_fab button { width:3.1rem; height:3.1rem; min-height:0; padding:0; border-radius:50%; border:none;
                          background:var(--ink); color:#FFFFFF; box-shadow:0 3px 10px rgba(30,42,51,0.28); }
.st-key-info_fab button p { font-size:1.3rem; font-weight:800; }
.st-key-info_fab button:hover { background:#2E3F4B; color:#FFFFFF; }
.st-key-info_fab button:focus-visible { outline:3px solid var(--approve); outline-offset:3px; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

CONDITION_NAMES = {
    "black_box": "Prediction only",
    "lime": "LIME explanation",
    "shap": "SHAP explanation",
    "plain_language": "Plain-language explanation",
}
APPROVE, REJECT = "#1F5FA8", "#C4661F"  # blue / orange: distinguishable with colour-vision deficiency
# Every scale point is named; answers are stored as their position (1 = first label).
AGREE = ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"]
CONFIDENCE = ["Not at all confident", "Slightly confident", "Moderately confident", "Very confident",
              "Completely confident"]
FAMILIARITY = ["Not at all familiar", "Slightly familiar", "Somewhat familiar", "Moderately familiar",
               "Extremely familiar"]
LENDING = ["No", "Yes, personally (e.g. my own loans)", "Yes, professionally"]
RANKS = ["Most useful", "Second most useful", "Least useful"]

# Adapted from the Trust in Automation scale (Jian, Bisantz & Drury, 2000).
TRUST_ITEMS = {
    "trust_1": "I can trust the system's recommendations.",
    "trust_2": "The system's recommendations are reliable.",
    "trust_3": "I am confident in the system's recommendations.",
    "trust_4": "I would rely on this system to help make loan decisions.",
    "trust_5_rev": "The system's recommendations could be misleading.",
    "trust_6_rev": "I am wary of this system.",
}
UNDERSTANDING_ITEM = {"understand_self": "I understand why the system made this recommendation."}
# Adapted from the Explanation Satisfaction Scale (Hoffman et al., 2018).
SATISFACTION_ITEMS = {
    "sat_1": "The explanation helps me understand how the system works.",
    "sat_2": "The explanation is satisfying.",
    "sat_3": "The explanation has sufficient detail.",
    "sat_4": "The explanation is easy to follow.",
}


@st.cache_resource
def load_explanations():
    return json.loads((ARTIFACTS / "explanations.json").read_text())


@st.cache_resource
def load_metrics():
    return json.loads((ARTIFACTS / "model_comparison.json").read_text())


EXP = load_explanations()
LABELS = EXP["labels"]


# ---------- rendering ----------

def applicant_panel(case):
    fields = "".join(f"<div><dt>{html.escape(LABELS[f])}</dt><dd>{html.escape(v)}</dd></div>"
                     for f, v in case["applicant"].items())
    st.markdown(f"<div class='sheet'><div class='sheet-head'><h3>Loan application</h3>"
                f"<span>Ref. {html.escape(case['id'])}</span></div><dl class='fields'>{fields}</dl></div>",
                unsafe_allow_html=True)


def prediction_card(case, show_outcome=False):
    """The system's assessment: decision, confidence and where the case sits against the 50% line."""
    approved, p = case["approved"], case["proba_approve"]
    kind = "approve" if approved else "reject"
    outcome = ""
    if show_outcome:
        correct = approved == case["actual_good"]
        outcome = (f"<p class='outcome'>Actual outcome: {'good' if case['actual_good'] else 'bad'} credit. "
                   f"<b class='{'' if correct else 'wrong'}'>The system was {'right' if correct else 'wrong'}.</b></p>")
    st.markdown(
        f"<div class='sheet assessment {kind}'>"
        f"<div class='sheet-head'><h3>System assessment</h3><span>{max(p, 1 - p):.0%} confident</span></div>"
        f"<p class='decision'>{kind.capitalize()}</p>"
        f"<div class='meter' role='img' aria-label='Estimated chance of approval {p:.0%}; "
        f"applications above 50% are approved'>"
        f"<div class='track'><span class='line' style='left:50%'></span>"
        f"<span class='dot' style='left:{p:.1%}'></span></div>"
        f"<div class='ends'><span>Reject</span><span>Approve</span></div>"
        f"<p>Estimated chance of approval: {p:.0%}. Applications above the line at 50% are approved.</p>"
        f"</div>{outcome}</div>", unsafe_allow_html=True)


def bar_chart(labels, weights, xlabel):
    labels, weights = labels[::-1], weights[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 0.4 * len(labels) + 0.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.barh(labels, weights, color=[APPROVE if w > 0 else REJECT for w in weights], height=0.62)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.tick_params(colors=INK, labelsize=10, length=0)
    ax.grid(axis="x", color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    fig.tight_layout()
    return fig


def lime_view(case, top=8):
    st.markdown("##### LIME explanation")
    st.caption("Each bar shows a condition about this applicant and how much it moved the system's "
               "local estimate. Blue bars support approval, orange bars support rejection.")
    rules = case["lime"][:top]
    fig = bar_chart([r["rule"] for r in rules], [r["weight"] for r in rules], "Weight towards approval")
    st.pyplot(fig)
    plt.close(fig)


def shap_view(case, top=8):
    st.markdown("##### SHAP explanation")
    st.caption("The system starts from the average applicant. "
               "Each bar shows how much a feature of this applicant moved that estimate up (blue) or down (orange).")
    items = sorted(case["shap"].items(), key=lambda kv: abs(kv[1]), reverse=True)
    shown, rest = items[:top], items[top:]
    labels = [f"{LABELS[f]} = {case['applicant'][f]}" for f, _ in shown]
    weights = [v for _, v in shown]
    if rest:
        labels.append(f"{len(rest)} other features")
        weights.append(sum(v for _, v in rest))
    if EXP["shap_units"].startswith("probability"):
        fig = bar_chart(labels, [w * 100 for w in weights], "Change in chance of approval (percentage points)")
        note = (f"Average applicant {EXP['shap_base']:.0%} + contributions = "
                f"{case['proba_approve']:.0%} chance of approval for this applicant.")
    else:
        fig = bar_chart(labels, weights, f"Contribution ({EXP['shap_units']})")
        note = f"Contributions are in {EXP['shap_units']}."
    st.pyplot(fig)
    plt.close(fig)
    st.caption(note)


def plain_language_view(case):
    st.markdown("##### Explanation")
    n = case["narrative"]
    st.markdown(n["headline"])
    st.markdown("The factors that influenced the system most:")
    st.markdown("\n".join(f"1. {r}" for r in n["reasons"]))
    st.markdown(n["summary"])


VIEWS = {"lime": lime_view, "shap": shap_view, "plain_language": plain_language_view}


def show_case(case, condition):
    form, decision = st.columns([1, 1], gap="large")
    with form:
        applicant_panel(case)
    with decision:
        prediction_card(case)
        if condition in VIEWS:
            VIEWS[condition](case)


# ---------- explore mode ----------

def explore(cid, compare, top, show_outcome):
    st.title("Loan application review", anchor=False)
    if cid is None:
        st.info("No applications match this filter. Choose another filter in the sidebar.")
        return
    case = EXP["cases"][cid]
    views = {"lime": lambda: lime_view(case, top), "shap": lambda: shap_view(case, top),
             "plain_language": lambda: plain_language_view(case)}
    form, decision = st.columns([1, 1], gap="large")
    with form:
        applicant_panel(case)
    with decision:
        prediction_card(case, show_outcome)
        if compare:
            views["plain_language"]()
        else:
            for tab, cond in zip(st.tabs(["LIME", "SHAP", "Plain language"]), views):
                with tab:
                    views[cond]()
    if compare:
        st.write("")
        lime_col, shap_col = st.columns(2, gap="large")
        with lime_col:
            views["lime"]()
        with shap_col:
            views["shap"]()

    st.write("")
    with st.expander("Model comparison and global feature importance"):
        m = load_metrics()
        st.markdown(f"Black-box model: **{m['chosen'].replace('_', ' ')}** (chosen by cross-validated ROC-AUC).")
        st.table({name.replace("_", " ").capitalize(): {**{f"CV {k}": v for k, v in r["cv"].items()},
                                                         **{f"Test {k}": v for k, v in r["test"].items()}}
                  for name, r in m["results"].items()})
        if (ARTIFACTS / "shap_global.png").exists():
            st.image(str(ARTIFACTS / "shap_global.png"),
                     caption="Average influence of each feature across the 200 test applicants (SHAP).")


# ---------- sidebar (Explore mode) ----------

FILTERS = {
    "All": lambda c: True,
    "Approved": lambda c: c["approved"],
    "Rejected": lambda c: not c["approved"],
    "Errors": lambda c: c["approved"] != c["actual_good"],
}


def hide_sidebar():
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl']"
                "{display:none}</style>", unsafe_allow_html=True)


def participant_link() -> str:
    """The study link for this deployment, e.g. https://<app>.streamlit.app/?mode=study."""
    base = (getattr(st.context, "url", None) or "http://localhost:8501").split("?")[0].rstrip("/")
    return f"{base}/?mode=study"


def researcher_password() -> str | None:
    try:
        return st.secrets.get("RESEARCHER_PASSWORD")
    except Exception:  # no secrets file locally
        return None


def researcher_unlocked() -> bool:
    """Explore mode shows every response, so online it sits behind a password."""
    password = researcher_password()
    if not password or st.session_state.get("researcher"):
        return True
    hide_sidebar()
    st.title("Researcher sign-in", anchor=False)
    st.markdown("This page is for the researcher. If you are taking part in the study, "
                "use the link you were sent.")
    with st.form("signin"):
        entered = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if hmac.compare_digest(entered.encode(), password.encode()):
                st.session_state["researcher"] = True
                st.rerun()
            st.error("Incorrect password. Check the RESEARCHER_PASSWORD secret and try again.")
    return False


def sidebar():
    """Researcher console: mode, case register, display options and study status."""
    sb = st.sidebar
    sb.markdown("## Loan review lab")
    sb.caption("XAI trust study, MSc in Artificial Intelligence, NCI")
    mode = sb.segmented_control("Mode", ["Explore", "Participant preview"], default="Explore",
                                key="mode", help="Participant preview runs the study exactly as participants see it.")
    mode = mode or "Explore"
    settings = {"mode": mode}

    if mode == "Explore":
        register = sb.container()
        sb.markdown("#### Display")
        settings["compare"] = sb.toggle("Compare all explanations", help="Show LIME, SHAP and plain language together.")
        settings["top"] = sb.slider("Factors shown in charts", 3, 8, 8)
        settings["show_outcome"] = sb.toggle("Show actual outcome", value=True,
                                             help="Reveal whether the applicant really had good credit.")

        with register:
            st.markdown("#### Applications")
            filters = list(FILTERS) if settings["show_outcome"] else list(FILTERS)[:3]
            chosen = st.segmented_control("Show", filters, default="All", key="filter",
                                          label_visibility="collapsed") or "All"
            if chosen not in filters:
                chosen = "All"
            ids = [cid for cid, c in EXP["cases"].items() if FILTERS[chosen](c)]

            def row(cid):
                c = EXP["cases"][cid]
                text = f"**Ref. {cid}** :gray[{'approved' if c['approved'] else 'rejected'}]"
                if settings["show_outcome"] and c["approved"] != c["actual_good"]:
                    text += " :orange[model wrong]"
                return text

            with st.container(key="register"):
                settings["cid"] = st.radio("Application", ids, format_func=row, key="case",
                                           label_visibility="collapsed") if ids else None

    sb.markdown("#### Study")
    total, done = storage.counts()
    sb.markdown(f"{total} participants started, {done} completed.")
    sb.caption(f"Participant link (responses saved to {storage.backend()})")
    sb.code(participant_link(), language=None)
    if total:
        sb.download_button("Download responses (CSV)", storage.export().to_csv(index=False),
                           "responses.csv", "text/csv", use_container_width=True)

    m = load_metrics()
    sb.caption(f"Model: {m['chosen'].replace('_', ' ')}, cross-validated ROC-AUC "
               f"{m['results'][m['chosen']]['cv']['roc_auc']:.2f}")
    return settings


# ---------- study mode ----------

def clear_answer(key):
    st.session_state[key] = None


class Questionnaire:
    """Numbered questions on one survey page; remembers which are still unanswered."""

    def __init__(self, page):
        self.page, self.n, self.missing = page, 0, []

    def section(self, title, note=None):
        note = f"<p>{html.escape(note)}</p>" if note else ""
        st.markdown(f"<div class='q-section'><h4>{html.escape(title)}</h4>{note}</div>", unsafe_allow_html=True)

    def _head(self, text):
        self.n += 1
        st.markdown(f"<div class='q-head'><span class='q-num'>{self.n}</span>"
                    f"<p class='q-text'>{html.escape(text)}</p></div>", unsafe_allow_html=True)

    def _key(self, variant=""):
        return f"q_{'first_' if self.n == 0 else ''}{variant}{self.page}_{self.n + 1}"

    def ask(self, text, options, key, format_func=str, variant="", on_change=None, clearable=False):
        with st.container(key=self._key(variant)):
            self._head(text)
            value = st.radio(text, options, index=None, key=key, format_func=format_func,
                             horizontal=True, label_visibility="collapsed", on_change=on_change)
            if clearable and value is not None:
                st.button("Clear", key=f"clear_{key}", type="tertiary", on_click=clear_answer, args=(key,),
                          help="Remove this answer")
        if value is None:
            self.missing.append(self.n)
        return value

    def scale(self, text, labels, key, variant=None, on_change=None, clearable=False):
        """A named rating scale; returns the chosen position (1 = first label) or None."""
        if variant is None:
            variant = "agree_" if labels is AGREE else ""
        value = self.ask(text, labels, key, variant=variant, on_change=on_change, clearable=clearable)
        return None if value is None else labels.index(value) + 1

    def comment(self, text, key):
        with st.container(key=self._key()):
            self._head(text)
            return st.text_area(text, key=key, label_visibility="collapsed")


def gated_button(label, problem, done="All questions answered.", align="right"):
    """A primary button that stays disabled while `problem` says what is missing."""
    with st.container(key="gate" if align == "right" else "gate_left"):
        if problem:
            st.markdown(f"<p class='gate'>{html.escape(problem)}</p>", unsafe_allow_html=True)
        elif done:
            st.markdown(f"<p class='gate ok'>{html.escape(done)}</p>", unsafe_allow_html=True)
        return st.button(label, type="primary", disabled=bool(problem), help=problem or None)


def missing_message(missing):
    return f"Answer all questions to continue ({len(missing)} left)." if missing else None


@st.dialog("About this study", width="large")
def study_information():
    st.markdown(study_text.INFORMATION)
    st.markdown("##### What you agreed to")
    st.markdown("\n".join(f"- {line}" for line in study_text.CONSENT_STATEMENTS))
    st.markdown(f"To stop taking part, close this page. Questions: {study_text.RESEARCHER_EMAIL} "
                f"(researcher) or {study_text.SUPERVISOR_EMAIL} (supervisor).")


def info_button():
    if st.button("?", key="info_fab", help="About this study: what it involves, your rights and who to contact"):
        study_information()


def short_name(label):
    return label.split(" (")[0]


def claim_rank_for(kind, kinds):
    """Keep ranks unique: if another explanation already holds the chosen rank, clear it there.
    Only reachable by keyboard, since taken ranks cannot be clicked (see lock_taken_ranks)."""
    def claim():
        chosen = st.session_state.get(f"rank_{kind}")
        for other in kinds:
            if other != kind and chosen is not None and st.session_state.get(f"rank_{other}") == chosen:
                st.session_state[f"rank_{other}"] = None
    return claim


def lock_taken_ranks(kinds):
    """Grey out, and block clicks on, each rank another explanation already holds, and say who holds it."""
    rules = []
    for kind in kinds:
        for other, label in kinds.items():
            taken = st.session_state.get(f"rank_{other}")
            if other == kind or taken is None:
                continue
            option = (f'[class*="st-key-q_"][class*="rank_{kind}_"] [data-testid="stRadioGroup"] '
                      f'> div:nth-child({RANKS.index(taken) + 1}) label')
            rules.append(f"{option} {{ opacity:0.5; pointer-events:none; cursor:not-allowed; }}")
            rules.append(f'{option} p::after {{ content:"Given to {short_name(label)}"; display:block; '
                         f"font-size:0.72rem; font-weight:400; }}")
    if rules:
        st.markdown(f"<style>{' '.join(rules)}</style>", unsafe_allow_html=True)


def factor_question(q, case, key):
    """'Which factor influenced the decision most?' Correct answer = top SHAP feature."""
    ranked = sorted(case["shap"], key=lambda f: abs(case["shap"][f]), reverse=True)
    rng = random.Random(case["id"])
    options = [ranked[0], *rng.sample(ranked[4:], 3)]
    rng.shuffle(options)
    answer = q.ask("Which factor do you think influenced the system's recommendation the most?",
                   options, key, format_func=LABELS.get)
    return {"top_factor_answer": answer, "top_factor_correct": answer == ranked[0]}


def study():
    s = st.session_state
    s.setdefault("step", "consent")

    if s.step == "consent":
        st.title("How do people judge AI loan decisions?", anchor=False)
        st.markdown(study_text.INFORMATION)
        st.markdown("##### Consent")
        st.markdown("\n".join(f"- {line}" for line in study_text.CONSENT_STATEMENTS))
        agreed = st.checkbox(study_text.CONSENT_CHECKBOX)
        if gated_button("Start", None if agreed else "Tick the consent box above to start.", done=None, align="left"):
            s.pid, s.order = storage.new_participant()
            s.block, s.step = 0, "background"
            st.rerun()

    elif s.step == "background":
        st.title("Before we begin", anchor=False)
        st.markdown("Two short questions about you. There are no right or wrong answers.")
        with st.container(key="sheet_background"):
            q = Questionnaire("background")
            ai = q.scale("How familiar are you with artificial intelligence or machine learning?",
                         FAMILIARITY, "bg_ai")
            loans = q.ask("Have you ever made or assessed lending or credit decisions?", LENDING, "bg_loans")
        if gated_button("Continue", missing_message(q.missing)):
            storage.save(s.pid, -1, "background", "", {"ai_familiarity": ai, "lending_experience": loans})
            s.step = "explained"
            st.rerun()

    elif s.step in ("explained", "forward"):
        block, condition = s.block, s.order[s.block]
        spec = EXP["blocks"][block]
        st.progress(block / 4, text=f"Part {block + 1} of 4")
        key = f"b{block}_{s.step}"

        if s.step == "explained":
            case = EXP["cases"][spec["explained"]]
            st.title(f"Part {block + 1}: review this decision", anchor=False)
            show_case(case, condition)
            with st.container(key=f"sheet_{key}"):
                q = Questionnaire(key)
                q.section("The decision")
                answers = factor_question(q, case, f"{key}_factor")
                q.section("Your view of the system", "How much do you agree with each statement?")
                answers |= {k: q.scale(text, AGREE, f"{key}_{k}") for k, text in (TRUST_ITEMS | UNDERSTANDING_ITEM).items()}
                if condition != "black_box":
                    q.section("The explanation", "How much do you agree with each statement about the explanation?")
                    answers |= {k: q.scale(text, AGREE, f"{key}_{k}") for k, text in SATISFACTION_ITEMS.items()}
            if gated_button("Next", missing_message(q.missing)):
                storage.save(s.pid, block, condition, case["id"], answers)
                s.step = "forward"
                st.rerun()
        else:
            case = EXP["cases"][spec["forward"]]
            st.title(f"Part {block + 1}: predict the system", anchor=False)
            form, intro = st.columns([1, 1], gap="large")
            with form:
                applicant_panel(case)
            with intro:
                st.markdown("Here is a **new** applicant. The system has assessed this application, but its "
                            "recommendation is hidden.\n\nRead the application, then answer the two questions "
                            "below: what do you think the system recommended, based on what you have seen so far?")
            with st.container(key=f"sheet_{key}"):
                q = Questionnaire(key)
                q.section("Your prediction")
                guess = q.ask("The system will recommend to...", ["Approve", "Reject"], f"{key}_guess")
                conf = q.scale("How confident are you in your prediction?", CONFIDENCE, f"{key}_conf")
            go = gated_button("Next", missing_message(q.missing))
            if go:
                storage.save(s.pid, block, condition, case["id"], {
                    "forward_guess": guess, "forward_confidence": conf,
                    "forward_correct": (guess == "Approve") == case["approved"]})
                s.block += 1
                s.step = "explained" if s.block < 4 else "final"
                st.rerun()

    elif s.step == "final":
        st.title("Last questions", anchor=False)
        st.markdown("You saw three kinds of explanation. Rank them from most to least useful.")
        kinds = {"lime": "LIME (bar chart of conditions)", "shap": "SHAP (bar chart of feature contributions)",
                 "plain-language": "Plain language (written sentences)"}
        lock_taken_ranks(kinds)
        with st.container(key="sheet_final"):
            q = Questionnaire("final")
            q.section("Usefulness of the explanations",
                      "Each rank can be given to one explanation only. To reuse a rank, clear it on the "
                      "explanation that has it first.")
            ranks = {k: q.scale(text, RANKS, f"rank_{k}", variant=f"rank_{k}_", on_change=claim_rank_for(k, kinds),
                                clearable=True)
                     for k, text in kinds.items()}
            q.section("Anything else?")
            comment = q.comment(study_text.COMMENT_LABEL, "comment")
        if gated_button("Submit my answers", missing_message(q.missing)):
            storage.save(s.pid, 4, "final", "", {**{f"rank_{k}": r for k, r in ranks.items()}, "comment": comment})
            storage.complete(s.pid)
            s.step = "done"
            st.rerun()

    else:
        st.title("Thank you!", anchor=False)
        st.markdown("Your responses have been saved.")
        st.markdown(study_text.DEBRIEF)
        st.markdown("You can now close this page.")

    info_button()


# Participants get a link ending in ?mode=study, which hides the sidebar and Explore mode.
if st.query_params.get("mode") == "study":
    hide_sidebar()
    study()
elif researcher_unlocked():
    opts = sidebar()
    if opts["mode"] == "Explore":
        explore(opts["cid"], opts["compare"], opts["top"], opts["show_outcome"])
    else:
        study()
