"""Participant-facing text. Must match Appendices A-C of the approved ethics application."""

RESEARCHER_EMAIL = "x25101056@student.ncirl.ie"
SUPERVISOR = "Syed Safi Uddin Qadri"
SUPERVISOR_EMAIL = "SyedSafiUddin.Qadri@ncirl.ie"

INFORMATION = f"""
This study is part of an MSc in Artificial Intelligence research project at the National College of Ireland,
by Mevin John Rejimon ({RESEARCHER_EMAIL}), supervised by {SUPERVISOR} ({SUPERVISOR_EMAIL}).

**What is the study about?** AI systems are increasingly used to help make decisions such as approving loans.
This study looks at whether explanations of an AI system's decisions help people understand those decisions and
decide how much to trust them.

**What will I be asked to do?** The study takes about **15-20 minutes**. You will see four loan applications, each
assessed by an AI system, sometimes with an explanation (a chart or a few sentences). After each one you answer
short questions about how much you trust and understand the system, and predict what it would decide for a new
applicant. At the end you rank the types of explanation. The system's recommendations may be right or wrong.
No knowledge of AI or finance is needed.

**Who can take part?** Anyone aged 18 or over who can read English.

**Do I have to take part?** No. Taking part is voluntary. You can stop at any time, without giving a reason and
without any penalty, by closing the page. Answers from unfinished sessions are not analysed and are deleted.
Because no identifying information is collected, your answers cannot be traced back to you, so a completed
response cannot be removed after you submit it.

**What information is collected?** Only your answers to the study questions, two background questions (your
familiarity with AI and any experience of lending decisions) and an optional comment. No names, email addresses or
other identifying details are collected; your answers are stored under a random code in an encrypted database in
the EU that only the researcher can access. The data is kept for five years in line with NCI's Research Data
Retention Policy and then deleted.

**Risks and benefits.** There are no foreseeable risks beyond normal computer use, and you will not be asked about
your own finances. The loan applications are anonymised historical records from a public research dataset, not
real current customers. There is no direct benefit to you, but your answers will help show how AI decisions can be
made easier to understand.

**Results.** Results will be reported in aggregate in an MSc practicum report and a presentation for examiners.
Optional comments may be quoted without any identifying detail. For a summary of the findings or any questions,
email {RESEARCHER_EMAIL}.
"""

CONSENT_STATEMENTS = [
    "I have read and understood the information about this study.",
    "I am aged 18 or over.",
    "I understand that taking part is voluntary and that I can stop at any time, without giving a reason, by "
    "closing the page.",
    "I understand that no identifying information is collected, and that my anonymous answers will be used in an "
    "MSc report and presentation.",
    "I understand that once I submit my answers they cannot be withdrawn, because they cannot be linked to me.",
]
CONSENT_CHECKBOX = "I confirm all of the above and agree to take part."

DEBRIEF = f"""
This study compares four ways of presenting an AI system's loan decision: no explanation, a LIME explanation, a
SHAP explanation and a written plain-language explanation. It tests whether explanations change how much people
trust and understand the system. The order of the four parts was varied between participants so that the order
does not affect the results.

**Some of the system's recommendations you saw were wrong.** The applications were chosen on purpose to include
both correct and incorrect decisions, because trusting an AI system well means relying on it when it is right and
questioning it when it is wrong. The system is a research prototype and was not used to make any real lending
decisions.

Your answers are anonymous and cannot be linked to you. If you would like a summary of the results or have any
questions, contact {RESEARCHER_EMAIL}.

If thinking about loans or money has raised any concerns for you, free and confidential support is available from
MABS, the Money Advice and Budgeting Service (helpline 0818 07 2000, www.mabs.ie). If you are feeling distressed
for any reason, Samaritans are available 24 hours a day on 116 123.
"""

COMMENT_LABEL = "Anything else you would like to tell us? (optional; please do not include names or personal details)"
