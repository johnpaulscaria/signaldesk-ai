
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

st.title("🎯 SignalDesk AI")
st.caption(
    "Multi-agent sales intelligence | "
    "Company research, evidence screening and sales strategy"
)

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
# 10. DASHBOARD
# =====================================================

with st.sidebar:
    st.header("Research settings")
    st.caption("Analyze a target account.")
    st.caption(f"Model: {MODEL}")

    if demo_mode:
        st.info("Demo mode: no Gemini API calls.")
    else:
        st.info("Live mode: cached Gemini requests enabled.")


with st.form("research_form"):
    company = st.text_input(
        "Target company",
        placeholder="e.g. Infosys"
    )
    industry = st.text_input(
        "Industry",
        value="IT services and data centers"
    )
    product = st.text_input(
        "Product or service you sell",
        value="Industrial energy audits"
    )

    submitted = st.form_submit_button(
        "Run Research",
        type="primary",
        use_container_width=True
    )


if submitted:
    if not company.strip() or not industry.strip() or not product.strip():
        st.warning("Complete all three fields.")

    elif demo_mode:
        st.session_state["latest_report"] = build_demo_report(
            company.strip(), industry.strip(), product.strip()
        )

    else:
        report = run_signaldesk(
            company.strip(), industry.strip(), product.strip()
        )
        st.session_state["latest_report"] = report


# =====================================================
# 11. RESULTS
# =====================================================

if "latest_report" in st.session_state:
    report = st.session_state["latest_report"]
    steps = report["steps"]

    if report["status"] == "failed":
        st.error("The workflow did not complete.")

        for error in report["errors"]:
            st.write(error)

        if any(
            marker in " ".join(report["errors"]).upper()
            for marker in ["429", "RESOURCE_EXHAUSTED", "QUOTA"]
        ):
            st.warning(
                "Gemini quota is exhausted. Enable Demo mode "
                "to explore the dashboard without API calls. "
                "Live research will work again when quota resets."
            )

    else:
        if report["status"] == "demo":
            st.warning(
                "DEMO MODE — all displayed research is illustrative. "
                "It is not verified information about the selected company."
            )
        else:
            st.success("Research report generated.")

        analyst = steps["company_analyst"]
        audit = steps["evidence_auditor"]
        sales = steps["sales_strategist"]
        sources = steps["public_sources"]

        gaps = audit.get("research_gaps", [])
        if isinstance(gaps, str):
            gaps = [gaps] if gaps.strip() else []

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Potential opportunities",
            len(analyst.get("opportunities", []))
        )
        col2.metric("Public news records", len(sources))
        col3.metric("Evidence gaps", len(gaps))

        tab1, tab2, tab3, tab4 = st.tabs([
            "Sales Strategy", "Opportunities", "Evidence", "Execution Log"
        ])

        with tab1:
            st.subheader("Recommended action")
            st.write(sales.get("next_action", ""))

            st.subheader("Opportunity rationale")
            st.write(sales.get("rationale", ""))

            st.subheader("Discovery questions")
            questions = sales.get("discovery_questions", [])
            if isinstance(questions, list):
                for question in questions:
                    st.markdown(f"- {question}")
            else:
                st.write(questions)

            st.subheader("Outreach draft")
            st.write(sales.get("outreach_draft", ""))
            st.caption(sales.get("evidence_caveat", ""))

        with tab2:
            for index, item in enumerate(
                analyst.get("opportunities", []), 1
            ):
                with st.expander(
                    f"{index}. {item.get('signal', 'Opportunity')}"
                ):
                    st.write("Business implication:", item.get("business_implication", ""))
                    st.write("Relevance score:", item.get("relevance_score", "N/A"), "/ 10")
                    st.write("Evidence gap:", item.get("evidence_gap", ""))
                    st.write("Next action:", item.get("next_research_action", ""))

        with tab3:
            st.subheader("Evidence quality")
            st.write(audit.get("overall_evidence_quality", "Not assessed"))

            st.subheader("Research gaps")
            for gap in gaps:
                st.markdown(f"- {gap}")

            st.subheader("Audited opportunities")
            for item in audit.get("audited_opportunities", []):
                st.markdown(f"**{item.get('signal', 'Signal')}**")
                st.write("Status:", item.get("evidence_status", "unassessed"))
                st.write(item.get("evidence_explanation", ""))

                for url in item.get("supporting_sources", []):
                    if isinstance(url, str) and url.startswith("https://"):
                        st.markdown(f"- [Open cited source]({url})")

                st.divider()

            st.subheader("Retrieved news")
            if sources:
                for source in sources:
                    st.markdown(f"- [{source['title']}]({source['url']})")
                    st.caption(
                        f"{source['publisher']} | {source['published']}"
                    )
            else:
                st.info("No news records retrieved. Treat all signals as hypotheses.")

        with tab4:
            st.write("Status:", report["status"])
            st.write("Started:", report["started_at"])
            st.write("Finished:", report.get("finished_at"))
            st.write("Warnings:", report["errors"])
            st.write("Completed stages:", list(steps.keys()))

            with st.expander("View full JSON report"):
                st.json(report)

        st.download_button(
            "Download full JSON report",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name="signaldesk_report.json",
            mime="application/json"
        )
