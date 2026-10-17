

import os
import json
import re
import hashlib
import xml.etree.ElementTree as ET
from urllib.parse import quote
from datetime import datetime, timezone

import requests
import streamlit as st
from google import genai


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SignalDesk AI",
    page_icon="🎯",
    layout="wide"
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

try:
    API_KEY = st.secrets.get(
        "GEMINI_API_KEY",
        os.getenv("GEMINI_API_KEY", "")
    )
except Exception:
    API_KEY = os.getenv("GEMINI_API_KEY", "")

# Custom visual system
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --sd-purple:#6d5ce8; --sd-ink:#20233a; --sd-muted:#747891; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f7f8fc; color: var(--sd-ink); }
.block-container { max-width: 1440px; padding-top: 1.4rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #ececf4; }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.sd-brand { display:flex; align-items:center; gap:12px; margin-bottom: 1.5rem; }
.sd-logo { background:linear-gradient(135deg,#7c6cf2,#5b8def); color:white; border-radius:14px; width:44px; height:44px; display:flex; align-items:center; justify-content:center; font-size:23px; box-shadow:0 7px 18px #7568e533; }
.sd-brand-name { font-family:Manrope,sans-serif; font-weight:800; font-size:1.45rem; letter-spacing:-.6px; color:#252641; }
.sd-subtitle { color:#7b7e96; font-size:.92rem; margin-top:-5px; }
.sd-eyebrow { text-transform:uppercase; letter-spacing:1.7px; font-size:.72rem; color:#7772c9; font-weight:700; }
.sd-title { font-family:Manrope,sans-serif; font-size:2.05rem; font-weight:800; letter-spacing:-1px; color:#20233a; margin:.15rem 0 .2rem; }
.sd-description { color:#777b93; margin-bottom:1.2rem; }
.sd-card { background:#fff; border:1px solid #ececf5; border-radius:18px; padding:20px 22px; box-shadow:0 5px 20px rgba(40,44,90,.035); height:100%; }
.sd-card-title { font-family:Manrope,sans-serif; font-size:1rem; font-weight:750; color:#282943; margin-bottom:9px; }
.sd-muted { color:#777b93; font-size:.88rem; }
.sd-agent-icon { width:42px; height:42px; display:flex; align-items:center; justify-content:center; border-radius:13px; background:#f0edff; color:#6d5ce8; font-size:21px; margin-bottom:12px; }
.sd-pill { display:inline-block; padding:4px 10px; border-radius:99px; background:#f0edff; color:#6553d8; font-size:.75rem; font-weight:700; }
.sd-opportunity { padding:13px 0; border-bottom:1px solid #f0f0f6; }
.sd-opportunity:last-child { border-bottom:0; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background:linear-gradient(100deg,#6d5ce8,#7d70ed); border:0; border-radius:11px; font-weight:700; min-height:44px; }
.stButton > button, .stDownloadButton > button { border-radius:10px; }
[data-testid="stMetric"] { background:#fff; border:1px solid #ececf5; padding:16px 18px; border-radius:16px; box-shadow:0 4px 16px rgba(40,44,90,.025); }
[data-testid="stMetricLabel"] { color:#777b93; }
[data-testid="stMetricValue"] { color:#282943; font-family:Manrope,sans-serif; font-weight:800; }
[data-baseweb="tab-list"] { gap:8px; }
[data-baseweb="tab"] { background:#fff; border-radius:10px 10px 0 0; padding:10px 16px; }
[data-testid="stExpander"] { background:#fff; border:1px solid #ececf5; border-radius:12px; }
hr { border-color:#ececf5; }
@media (max-width: 800px) { .sd-title {font-size:1.6rem;} .sd-card {padding:16px;} }
</style>
""")

# Demo mode requires no API calls.
demo_mode = st.toggle(
    "Demo mode — use sample data without calling Gemini",
    value=False,
    help="Use this mode to demonstrate the dashboard without consuming API quota."
)

if not demo_mode and not API_KEY:
    st.warning(
        "Gemini API key not configured. Enable Demo mode "
        "or configure GEMINI_API_KEY in Streamlit Secrets."
    )
    st.stop()


# =====================================================
# 2. CACHED AI CALLS
# =====================================================

@st.cache_data(ttl=86400, max_entries=100)
def cached_ai_response(prompt, model, key_fingerprint):
    """
    Cache identical prompts for 24 hours.
    The API key itself is not stored in the cache key;
    only its fingerprint is used.
    """
    api_key = st.session_state.get("_signaldesk_api_key")

    if not api_key:
        raise RuntimeError("Gemini API key is unavailable.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    if not response.text:
        raise ValueError("The AI model returned no text.")

    return response.text


def ask_agent(prompt):
    if not API_KEY:
        raise RuntimeError("Gemini API key is missing.")

    # Store the key for the duration of this session.
    st.session_state["_signaldesk_api_key"] = API_KEY

    fingerprint = hashlib.sha256(
        API_KEY.encode()
    ).hexdigest()[:16]

    return cached_ai_response(
        prompt, MODEL, fingerprint
    )


def parse_json(text):
    text = (text or "").strip()
    text = re.sub(
        r"^```(?:json)?\s*", "", text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for opening, closing in [("{", "}"), ("[", "]")]:
            start = text.find(opening)
            end = text.rfind(closing)

            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])

        raise ValueError("AI response was not valid JSON.")


def ask_agent_json(prompt):
    return parse_json(ask_agent(prompt))


# =====================================================
# 3. PUBLIC NEWS RETRIEVAL
# =====================================================

@st.cache_data(ttl=3600, max_entries=100)
def fetch_public_evidence(company, industry, limit=10):
    query = quote(f'"{company}" {industry}')

    feeds = [
        (
            "Google News",
            "https://news.google.com/rss/search?q="
            + query + "&hl=en-IN&gl=IN&ceid=IN:en"
        ),
        (
            "Bing News",
            "https://www.bing.com/news/search?q="
            + query + "&format=rss"
        )
    ]

    errors = []

    for provider, url in feeds:
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 SignalDeskAI/1.0"},
                timeout=15
            )
            response.raise_for_status()

            root = ET.fromstring(response.content)
            articles = []

            for item in root.findall(".//item"):
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                published = item.findtext("pubDate", "").strip()
                source = item.find("source")

                if title and link:
                    articles.append({
                        "title": title,
                        "url": link,
                        "published": published,
                        "publisher": (
                            source.text.strip()
                            if source is not None and source.text
                            else provider
                        )
                    })

                if len(articles) >= limit:
                    break

            if articles:
                return articles

            errors.append(f"{provider}: no articles returned")

        except Exception as exc:
            errors.append(
                f"{provider}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError("News retrieval failed: " + " | ".join(errors))


# =====================================================
# 4. AGENT 1 — SIGNAL HUNTER
# =====================================================

def signal_hunter(company, industry, product):
    prompt = f"""
    You are Agent 1: Signal Hunter.

    Company: {company}
    Industry: {industry}
    Product sold: {product}

    Suggest five potential buying signals to investigate.
    These are hypotheses, not verified company facts.

    Return a JSON array of objects containing:
    signal, why_it_matters, evidence_needed, confidence.

    Do not invent company news or events.
    """

    result = ask_agent_json(prompt)

    if not isinstance(result, list):
        raise ValueError("Signal Hunter must return a JSON array.")

    return result


# =====================================================
# 5. AGENT 2 — COMPANY ANALYST
# =====================================================

def company_analyst(company, industry, product, signals):
    prompt = f"""
    You are Agent 2: Company Analyst.

    Company: {company}
    Industry: {industry}
    Product: {product}

    Potential signals:
    {json.dumps(signals, ensure_ascii=False)}

    Analyze business relevance. Score each opportunity from
    1 to 10, explain its possible implication, identify the
    evidence gap and recommend a research action.

    A relevance score is not purchase probability.
    Treat all unverified signals as hypotheses.

    Return a JSON object containing:
    company, product_sold, opportunities,
    overall_assessment, recommended_next_step.

    Each opportunity should contain:
    signal, business_implication, relevance_score,
    evidence_status, evidence_gap, next_research_action.
    """

    return ask_agent_json(prompt)


# =====================================================
# 6. AGENT 3 — EVIDENCE AUDITOR
# =====================================================

def evidence_auditor(company, analyst, sources):
    prompt = f"""
    You are Agent 3: Evidence Auditor.

    Company: {company}

    Analyst report:
    {json.dumps(analyst, ensure_ascii=False)}

    Retrieved public sources:
    {json.dumps(sources, ensure_ascii=False)}

    Audit each opportunity using only these source records.

    Allowed evidence statuses:
    supported, partially_supported, unsupported,
    no_relevant_source.

    Return a JSON object containing:
    company, audited_opportunities,
    overall_evidence_quality, research_gaps,
    recommended_follow_up.

    Each audited opportunity must contain:
    signal, evidence_status, supporting_sources,
    evidence_explanation, unsupported_assumptions,
    follow_up_needed.

    Copy source URLs exactly from the supplied records.
    A headline is not proof of buying intent.
    Do not claim to have read full articles.
    If there are no sources, use no_relevant_source.
    """

    return ask_agent_json(prompt)


# =====================================================
# 7. AGENT 4 — SALES STRATEGIST
# =====================================================

def sales_strategist(company, industry, product, audit):
    prompt = f"""
    You are Agent 4: Sales Strategist.

    Company: {company}
    Industry: {industry}
    Product: {product}

    Final evidence audit:
    {json.dumps(audit, ensure_ascii=False)}

    Return a JSON object containing:
    company, opportunity_priority, rationale,
    discovery_questions, next_action,
    outreach_draft, evidence_caveat.

    Prioritize evidence-backed opportunities.
    If evidence is weak, recommend more research.
    Never invent company facts or contact details.
    Do not contact anyone or send messages.
    """

    return ask_agent_json(prompt)


# =====================================================
# 8. DEMO DATA — ZERO API CALLS
# =====================================================

def build_demo_report(company, industry, product):
    sources = [
        {
            "title": "Sample source: Industrial efficiency initiative",
            "url": "https://example.com/sample-industrial-efficiency",
            "published": "Sample date — replace with a real article",
            "publisher": "Demonstration data"
        },
        {
            "title": "Sample source: Facility expansion announcement",
            "url": "https://example.com/sample-facility-expansion",
            "published": "Sample date — replace with a real article",
            "publisher": "Demonstration data"
        }
    ]

    signals = [
        {
            "signal": "Potential facility expansion",
            "why_it_matters": "A new facility could create an opportunity to review energy use.",
            "evidence_needed": "A verified company announcement.",
            "confidence": "low"
        },
        {
            "signal": "Potential energy-efficiency initiative",
            "why_it_matters": "Efficiency programs may create demand for audits.",
            "evidence_needed": "A published company target or initiative.",
            "confidence": "low"
        }
    ]

    opportunities = [
        {
            "signal": "Potential facility expansion",
            "business_implication": "A new facility could require energy performance assessment.",
            "relevance_score": 7,
            "evidence_status": "unverified",
            "evidence_gap": "No real company announcement has been checked.",
            "next_research_action": "Search official company releases."
        },
        {
            "signal": "Potential energy-efficiency initiative",
            "business_implication": "An efficiency program could create a need for energy audits.",
            "relevance_score": 6,
            "evidence_status": "unverified",
            "evidence_gap": "No verified company target has been checked.",
            "next_research_action": "Review the company's sustainability report."
        }
    ]

    audit = {
        "company": company,
        "audited_opportunities": [
            {
                "signal": item["signal"],
                "evidence_status": "no_relevant_source",
                "supporting_sources": [],
                "evidence_explanation": (
                    "Demonstration only. No real evidence "
                    "was retrieved or verified."
                ),
                "unsupported_assumptions": (
                    "The company may have this need."
                ),
                "follow_up_needed": item["evidence_gap"]
            }
            for item in opportunities
        ],
        "overall_evidence_quality": "demo_only",
        "research_gaps": [
            "Verify company announcements.",
            "Find evidence of a relevant business need."
        ],
        "recommended_follow_up": (
            "Review official company sources before outreach."
        )
    }

    sales = {
        "company": company,
        "opportunity_priority": "Unverified — research first",
        "rationale": (
            "These illustrative opportunities are not verified "
            "facts about the target company."
        ),
        "discovery_questions": [
            "What energy-efficiency objectives are currently in place?",
            "Are there new facilities or major upgrades planned?",
            "How is energy performance currently measured?"
        ],
        "next_action": (
            "Research official company announcements before outreach."
        ),
        "outreach_draft": (
            f"Hello, I work with organizations on {product}. "
            "I would be interested in learning how your team "
            "approaches energy performance and efficiency."
        ),
        "evidence_caveat": (
            "DEMO DATA — not verified research about this company."
        )
    }

    return {
        "company": company,
        "industry": industry,
        "product_sold": product,
        "status": "demo",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {
            "signal_hunter": signals,
            "company_analyst": {
                "company": company,
                "product_sold": product,
                "opportunities": opportunities,
                "overall_assessment": "Illustrative demo only.",
                "recommended_next_step": "Verify public evidence."
            },
            "public_sources": sources,
            "evidence_auditor": audit,
            "sales_strategist": sales
        },
        "errors": []
    }


# =====================================================
# 9. WORKFLOW ORCHESTRATOR
# =====================================================

def run_signaldesk(company, industry, product):
    report = {
        "company": company,
        "industry": industry,
        "product_sold": product,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {},
        "errors": []
    }

    try:
        with st.status("Running four specialist agents...", expanded=True) as status:

            st.write("1/4 — Signal Hunter")
            signals = signal_hunter(company, industry, product)
            report["steps"]["signal_hunter"] = signals

            st.write("2/4 — Company Analyst")
            analyst = company_analyst(
                company, industry, product, signals
            )
            report["steps"]["company_analyst"] = analyst

            st.write("Retrieving public news")
            try:
                sources = fetch_public_evidence(company, industry)
            except Exception as exc:
                sources = []
                report["errors"].append(str(exc))

            report["steps"]["public_sources"] = sources

            st.write("3/4 — Evidence Auditor")
            audit = evidence_auditor(company, analyst, sources)
            report["steps"]["evidence_auditor"] = audit

            st.write("4/4 — Sales Strategist")
            sales = sales_strategist(
                company, industry, product, audit
            )
            report["steps"]["sales_strategist"] = sales

            report["status"] = "completed"
            status.update(label="Workflow completed", state="complete")

    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(
            f"{type(exc).__name__}: {exc}"
        )

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    return report


# =====================================================
# 10. DASHBOARD — REDESIGNED UI
# =====================================================

with st.sidebar:
    st.markdown("""
    <div class="sd-brand">
      <div class="sd-logo">↗</div>
      <div><div class="sd-brand-name">SignalDesk AI</div><div class="sd-muted">Sales intelligence</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**WORKSPACE**")
    section = st.radio(
        "Navigation",
        ["Dashboard", "Company Research", "Agent Workflow", "Opportunities", "Evidence & Sources", "Reports"],
        label_visibility="collapsed",
        index=0
    )
    st.divider()
    st.markdown("**RESEARCH SETTINGS**")
    st.caption(f"Model: `{MODEL}`")
    if demo_mode:
        st.success("Demo mode · No API calls")
    else:
        st.info("Live mode · Gemini API")
    st.caption("Research outputs are decision support, not verified buying intent.")

st.markdown('<div class="sd-eyebrow">AI-POWERED ACCOUNT RESEARCH</div>', unsafe_allow_html=True)
st.markdown('<div class="sd-title">Find the signal behind every sale.</div>', unsafe_allow_html=True)
st.markdown('<div class="sd-description">Research target accounts, evaluate potential opportunities, and plan a more informed first conversation.</div>', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="sd-card-title">⌕ &nbsp; Research a company</div>', unsafe_allow_html=True)
    with st.form("research_form"):
        c1, c2 = st.columns([1, 1])
        with c1:
            company = st.text_input("Target company", placeholder="e.g. Infosys")
        with c2:
            industry = st.text_input("Industry", value="IT services and data centers")
        product = st.text_input("What are you selling?", value="Industrial energy audits", placeholder="Product or service")
        submitted = st.form_submit_button("✦  Run company analysis", type="primary", use_container_width=True)

if submitted:
    if not company.strip() or not industry.strip() or not product.strip():
        st.warning("Complete all three fields to start research.")
    elif demo_mode:
        st.session_state["latest_report"] = build_demo_report(company.strip(), industry.strip(), product.strip())
    else:
        st.session_state["latest_report"] = run_signaldesk(company.strip(), industry.strip(), product.strip())

if "latest_report" not in st.session_state:
    st.markdown("### Your research workspace")
    a, b, c, d = st.columns(4)
    cards = [
        (a, "⌕", "Research agent", "Finds potential company signals"),
        (b, "◈", "Analysis agent", "Connects signals to your offer"),
        (c, "✓", "Evidence auditor", "Checks available source records"),
        (d, "◎", "Sales strategist", "Suggests next steps and outreach"),
    ]
    for col, icon, title, desc in cards:
        with col:
            st.markdown(f'<div class="sd-card"><div class="sd-agent-icon">{icon}</div><div class="sd-card-title">{title}</div><div class="sd-muted">{desc}</div><br><span class="sd-pill">Ready to run</span></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="sd-card"><div class="sd-card-title">How SignalDesk works</div><p class="sd-muted">One research request passes through four specialist stages. Each stage adds context before the final sales recommendation is produced.</p><div class="sd-opportunity"><b>01 &nbsp; Discover</b><br><span class="sd-muted">Generate buying-signal hypotheses to investigate.</span></div><div class="sd-opportunity"><b>02 &nbsp; Analyze</b><br><span class="sd-muted">Assess relevance to your product or service.</span></div><div class="sd-opportunity"><b>03 &nbsp; Audit</b><br><span class="sd-muted">Compare opportunities with retrieved public news records.</span></div><div class="sd-opportunity"><b>04 &nbsp; Recommend</b><br><span class="sd-muted">Create discovery questions and a suggested next action.</span></div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="sd-card"><div class="sd-card-title">Built for better prospecting</div><div class="sd-opportunity"><b>Less manual searching</b><br><span class="sd-muted">Start with a structured company brief instead of a blank page.</span></div><div class="sd-opportunity"><b>Evidence-aware research</b><br><span class="sd-muted">See when source records are missing or don’t support a claim.</span></div><div class="sd-opportunity"><b>Practical sales outputs</b><br><span class="sd-muted">Get opportunity notes, discovery questions and an outreach draft.</span></div><br><span class="sd-pill">Prototype · Work in progress</span></div>', unsafe_allow_html=True)

if "latest_report" in st.session_state:
    report = st.session_state["latest_report"]
    steps = report.get("steps", {})
    if report.get("status") == "failed":
        st.error("The workflow did not complete.")
        for error in report.get("errors", []):
            st.write(error)
        if any(marker in " ".join(report.get("errors", [])).upper() for marker in ["429", "RESOURCE_EXHAUSTED", "QUOTA"]):
            st.warning("Gemini quota is exhausted. Enable Demo mode to explore the interface without API calls. Live research can resume when quota is available.")
    else:
        is_demo = report.get("status") == "demo"
        if is_demo:
            st.warning("DEMO MODE — the examples below are illustrative, not verified facts about the selected company.")
        else:
            st.success("Research workflow completed. Review the evidence before using findings in outreach.")

        analyst = steps.get("company_analyst", {})
        audit = steps.get("evidence_auditor", {})
        sales = steps.get("sales_strategist", {})
        sources = steps.get("public_sources", [])
        opportunities = analyst.get("opportunities", [])
        gaps = audit.get("research_gaps", [])
        if isinstance(gaps, str):
            gaps = [gaps] if gaps.strip() else []
        audited = audit.get("audited_opportunities", [])

        st.markdown(f'<div class="sd-eyebrow">ACCOUNT OVERVIEW</div><div class="sd-title">{report.get("company", "Company research")}</div><div class="sd-description">{report.get("industry", "")} &nbsp; · &nbsp; Offering: {report.get("product_sold", "")}</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Potential opportunities", len(opportunities))
        m2.metric("Public news records", len(sources))
        m3.metric("Research gaps", len(gaps))
        supported_count = sum(1 for x in audited if x.get("evidence_status") == "supported")
        m4.metric("Supported findings", supported_count)

        st.markdown("### Agent workflow")
        agent_cols = st.columns(4)
        agent_data = [
            ("⌕", "Research agent", "Potential signals", "signal_hunter" in steps),
            ("◈", "Analysis agent", "Business relevance", "company_analyst" in steps),
            ("✓", "Evidence auditor", "Source screening", "evidence_auditor" in steps),
            ("◎", "Sales strategist", "Next-step planning", "sales_strategist" in steps),
        ]
        for col, (ico, name, desc, complete) in zip(agent_cols, agent_data):
            with col:
                status_text = "Completed" if complete else "Not run"
                st.markdown(f'<div class="sd-card"><div class="sd-agent-icon">{ico}</div><div class="sd-card-title">{name}</div><div class="sd-muted">{desc}</div><br><span class="sd-pill">{"✓ " if complete else "○ "}{status_text}</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab_overview, tab_opportunities, tab_evidence, tab_strategy, tab_log = st.tabs(["Overview", "Opportunities", "Evidence & Sources", "Sales Strategy", "Execution Log"])
        with tab_overview:
            left, right = st.columns([1.15, 1])
            with left:
                st.markdown('<div class="sd-card-title">Top opportunities</div>', unsafe_allow_html=True)
                if opportunities:
                    for idx, item in enumerate(sorted(opportunities, key=lambda x: float(x.get("relevance_score", 0) or 0), reverse=True)[:5], 1):
                        score = item.get("relevance_score", "N/A")
                        st.markdown(f'<div class="sd-opportunity"><b><span style="color:#6d5ce8">{idx:02d}</span> &nbsp; {item.get("signal", "Potential opportunity")}</b><br><span class="sd-muted">{item.get("business_implication", "Potential relevance to your offering.")}</span><br><span class="sd-pill">Relevance: {score}/10</span></div>', unsafe_allow_html=True)
                else:
                    st.info("No opportunity records in this report.")
            with right:
                st.markdown('<div class="sd-card-title">Sales recommendation</div>', unsafe_allow_html=True)
                st.write(sales.get("next_action", "No next action provided."))
                st.markdown("**Why this approach?**")
                st.write(sales.get("rationale", "No rationale provided."))
                st.markdown("**Evidence caveat**")
                st.caption(sales.get("evidence_caveat", "Treat unverified signals as hypotheses."))
            st.markdown("### Recent public records")
            if sources:
                for source in sources[:5]:
                    st.markdown(f"- [{source.get('title', 'Open source')}]({source.get('url', '#')})")
                    st.caption(f"{source.get('publisher', 'Publisher unavailable')} · {source.get('published', 'Date unavailable')}")
            else:
                st.info("No public news records were retrieved. Treat signals as hypotheses.")

        with tab_opportunities:
            if not opportunities:
                st.info("No opportunities to display.")
            for index, item in enumerate(opportunities, 1):
                with st.expander(f"{index}. {item.get('signal', 'Potential opportunity')}  ·  Score {item.get('relevance_score', 'N/A')}/10"):
                    st.markdown("**Business implication**")
                    st.write(item.get("business_implication", "Not provided."))
                    st.markdown("**Evidence gap**")
                    st.write(item.get("evidence_gap", "Not provided."))
                    st.markdown("**Suggested research action**")
                    st.write(item.get("next_research_action", "Not provided."))
                    st.caption("Relevance score is not purchase probability.")

        with tab_evidence:
            st.markdown("### Evidence quality")
            st.write(audit.get("overall_evidence_quality", "Not assessed"))
            st.markdown("### Research gaps")
            if gaps:
                for gap in gaps:
                    st.markdown(f"- {gap}")
            else:
                st.caption("No research gaps were returned by the auditor.")
            st.markdown("### Audited opportunities")
            for item in audited:
                status = item.get("evidence_status", "unassessed")
                st.markdown(f"**{item.get('signal', 'Signal')}** &nbsp; · &nbsp; `{status}`")
                st.write(item.get("evidence_explanation", "No explanation provided."))
                if item.get("unsupported_assumptions"):
                    st.caption("Assumptions: " + str(item["unsupported_assumptions"]))
                for url in item.get("supporting_sources", []):
                    if isinstance(url, str) and url.startswith("https://"):
                        st.markdown(f"[Open cited source]({url})")
                st.divider()
            st.markdown("### Retrieved public news")
            if sources:
                for source in sources:
                    st.markdown(f"**[{source.get('title', 'Open source')}]({source.get('url', '#')})**")
                    st.caption(f"{source.get('publisher', 'Publisher unavailable')} · {source.get('published', 'Date unavailable')}")
            else:
                st.info("No news records retrieved. Do not treat generated signals as verified facts.")

        with tab_strategy:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown('<div class="sd-card-title">Recommended next action</div>', unsafe_allow_html=True)
                st.write(sales.get("next_action", "Not provided."))
                st.markdown('<div class="sd-card-title">Rationale</div>', unsafe_allow_html=True)
                st.write(sales.get("rationale", "Not provided."))
            with c2:
                st.markdown('<div class="sd-card-title">Discovery questions</div>', unsafe_allow_html=True)
                questions = sales.get("discovery_questions", [])
                if isinstance(questions, list):
                    for question in questions:
                        st.markdown(f"- {question}")
                else:
                    st.write(questions)
            st.markdown("### Outreach draft")
            st.text_area("Review and edit before using", value=str(sales.get("outreach_draft", "")), height=160, key="outreach_review")
            st.caption(sales.get("evidence_caveat", "Verify relevant facts before outreach."))

        with tab_log:
            st.markdown("### Execution summary")
            st.write("Status:", report.get("status"))
            st.write("Started:", report.get("started_at"))
            st.write("Finished:", report.get("finished_at", "Not recorded"))
            st.write("Warnings:", report.get("errors", []))
            st.write("Completed stages:", list(steps.keys()))
            with st.expander("View full JSON report"):
                st.json(report)

        st.divider()
        st.download_button("↓  Download full JSON report", data=json.dumps(report, indent=2, ensure_ascii=False), file_name="signaldesk_report.json", mime="application/json", type="primary")
