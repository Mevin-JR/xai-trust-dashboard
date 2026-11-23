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

from xai import storage
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
SCALE = ["1 - Strongly disagree", "2", "3", "4", "5", "6", "7 - Strongly agree"]

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

def score(choice):
    """Store scale answers as numbers (e.g. "7 - Strongly agree" -> 7)."""
    return None if choice is None else int(choice[0])


def likert(items, key):
    return {k: score(st.radio(text, SCALE, index=None, horizontal=True, key=f"{key}_{k}"))
            for k, text in items.items()}


def factor_question(case, key):
    """'Which factor influenced the decision most?' Correct answer = top SHAP feature."""
    ranked = sorted(case["shap"], key=lambda f: abs(case["shap"][f]), reverse=True)
    rng = random.Random(case["id"])
    options = [ranked[0], *rng.sample(ranked[4:], 3)]
    rng.shuffle(options)
    answer = st.radio("Which factor do you think influenced the system's recommendation the most?",
                      options, format_func=LABELS.get, index=None, key=key)
    return {"top_factor_answer": answer, "top_factor_correct": answer == ranked[0]}


def study():
    s = st.session_state
    s.setdefault("step", "consent")

    if s.step == "consent":
        st.title("How do people judge AI loan decisions?", anchor=False)
        st.markdown("""
This study is part of an MSc research project at the National College of Ireland. You will see
loan applications assessed by an AI system, sometimes with an explanation, and answer short
questions about them. It takes about **15-20 minutes**.

- Participation is voluntary and you can stop at any time.
- No names, emails or other personal details are collected. Responses are stored under a random ID.
- The applicants are from a public research dataset; they are not real customers.
""")
        if st.checkbox("I have read the information above and agree to take part.") and st.button("Start", type="primary"):
            s.pid, s.order = storage.new_participant()
            s.block, s.step = 0, "background"
            st.rerun()

    elif s.step == "background":
        st.title("Before we begin", anchor=False)
        ai = st.radio("How familiar are you with artificial intelligence or machine learning?",
                      ["1 - Not at all", "2", "3", "4", "5 - Very familiar"], index=None, horizontal=True)
        loans = st.radio("Have you ever made or assessed lending or credit decisions?",
                         ["No", "Yes, personally (e.g. my own loans)", "Yes, professionally"], index=None)
        if st.button("Continue", type="primary", disabled=None in (ai, loans)):
            storage.save(s.pid, -1, "background", "", {"ai_familiarity": score(ai), "lending_experience": loans})
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
            st.divider()
            answers = factor_question(case, f"{key}_factor")
            answers |= likert(TRUST_ITEMS | UNDERSTANDING_ITEM, key)
            if condition != "black_box":
                answers |= likert(SATISFACTION_ITEMS, key)
            if st.button("Next", type="primary", disabled=None in answers.values()):
                storage.save(s.pid, block, condition, case["id"], answers)
                s.step = "forward"
                st.rerun()
        else:
            case = EXP["cases"][spec["forward"]]
            st.title(f"Part {block + 1}: predict the system", anchor=False)
            form, questions = st.columns([1, 1], gap="large")
            with form:
                applicant_panel(case)
            with questions:
                st.markdown("Here is a **new** applicant. Based on what you have seen so far, "
                            "what do you think the system will recommend?")
                guess = st.radio("The system will recommend to...", ["Approve", "Reject"], index=None,
                                 horizontal=True, key=f"{key}_guess")
                conf = st.radio("How confident are you in your prediction?", SCALE, index=None,
                                horizontal=True, key=f"{key}_conf")
                go = st.button("Next", type="primary", disabled=None in (guess, conf))
            if go:
                storage.save(s.pid, block, condition, case["id"], {
                    "forward_guess": guess, "forward_confidence": score(conf),
                    "forward_correct": (guess == "Approve") == case["approved"]})
                s.block += 1
                s.step = "explained" if s.block < 4 else "final"
                st.rerun()

    elif s.step == "final":
        st.title("Last questions", anchor=False)
        st.markdown("You saw three kinds of explanation. Rank them from most to least useful.")
        kinds = ["LIME (bar chart of conditions)", "SHAP (bar chart of feature contributions)",
                 "Plain-language (written sentences)"]
        ranks = {k: st.selectbox(k, ["1 - most useful", "2", "3 - least useful"], index=None, key=f"rank_{k}")
                 for k in kinds}
        valid = None not in ranks.values() and len(set(ranks.values())) == 3
        if None not in ranks.values() and not valid:
            st.warning("Please give each explanation a different rank.")
        comment = st.text_area("Anything else you would like to tell us? (optional)")
        if st.button("Submit", type="primary", disabled=not valid):
            storage.save(s.pid, 4, "final", "", {**{f"rank_{k.split()[0].lower().rstrip('-')}": score(v) for k, v in ranks.items()},
                                                  "comment": comment})
            storage.complete(s.pid)
            s.step = "done"
            st.rerun()

    else:
        st.title("Thank you!", anchor=False)
        st.markdown("Your responses have been saved. You can now close this page.")


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
