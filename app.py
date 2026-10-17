
import json
import os
import re
from datetime import datetime, timezone
from urllib.parse import quote
import xml.etree.ElementTree as ET

import requests
import streamlit as st
from google import genai


# =====================================================
# 1. APP CONFIGURATION
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
    "Multi-agent B2B sales intelligence | "
    "Company research, evidence screening and sales strategy"
)

if not API_KEY:
    st.error(
        "Gemini API key not configured. Add "
        "GEMINI_API_KEY to Streamlit Secrets."
    )
    st.stop()

client = genai.Client(api_key=API_KEY)


# =====================================================
# 2. GENERAL AI AND JSON UTILITIES
# =====================================================

def ask_agent(prompt):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    if not response.text:
        raise ValueError("The AI model returned an empty response.")

    return response.text


def parse_json(text):
    """Parse JSON, including responses wrapped in code fences."""
    text = (text or "").strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])

        start = text.find("[")
        end = text.rfind("]")

        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])

        raise ValueError(
            "The AI response was not valid JSON."
        )


def ask_agent_json(prompt):
    return parse_json(ask_agent(prompt))


# =====================================================
# 3. PUBLIC NEWS RESEARCH
# =====================================================

def fetch_public_evidence(company, industry, limit=10):
    """
    Retrieve public news headlines and source URLs.
    Try Google News RSS, then Bing News RSS.
    """

    query = quote(f'"{company}" {industry}')

    feeds = [
        (
            "Google News",
            "https://news.google.com/rss/search?q="
            + query
            + "&hl=en-IN&gl=IN&ceid=IN:en"
        ),
        (
            "Bing News",
            "https://www.bing.com/news/search?q="
            + query
            + "&format=rss"
        )
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 SignalDeskAI/1.0"
    }

    errors = []

    for provider, url in feeds:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=20
            )
            response.raise_for_status()

            root = ET.fromstring(response.content)
            results = []

            for item in root.findall(".//item"):
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                published = item.findtext("pubDate", "").strip()

                source_element = item.find("source")
                publisher = (
                    source_element.text.strip()
                    if source_element is not None
                    and source_element.text
                    else provider
                )

                if title and link:
                    results.append({
                        "title": title,
                        "url": link,
                        "published": published,
                        "publisher": publisher,
                        "retrieved_from": provider
                    })

                if len(results) >= limit:
                    break

            if results:
                return results

            errors.append(
                f"{provider}: feed returned no usable articles"
            )

        except Exception as exc:
            errors.append(
                f"{provider}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "No public news could be retrieved. "
        + " | ".join(errors)
    )


# =====================================================
# 4. AGENT 1 — SIGNAL HUNTER
# =====================================================

def signal_hunter(company, industry, product):
    prompt = f"""
    You are Agent 1: Signal Hunter.

    Target company: {company}
    Industry: {industry}
    Product sold: {product}

    Identify five potential business events that could
    create a buying opportunity for this product.

    Consider expansion, investment, hiring, new facilities,
    operational challenges, regulations and public initiatives.

    IMPORTANT:
    You have not been given verified company research.
    Do not claim that any suggested event actually happened.
    Return potential signals as hypotheses.

    Return a JSON array with:
    signal, why_it_matters, evidence_needed,
    confidence.

    confidence must be low, medium or high.
    """

    result = ask_agent_json(prompt)

    if not isinstance(result, list):
        raise ValueError(
            "Signal Hunter did not return a JSON array."
        )

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

    Potential signals from Agent 1:
    {json.dumps(signals, ensure_ascii=False)}

    For each signal:
    1. Explain the possible business implication.
    2. Score relevance from 1 to 10.
    3. Identify missing evidence.
    4. Recommend a research action.

    These are hypotheses, not verified company facts.
    Relevance is NOT probability of purchase.

    Return a JSON object:
    {{
      "company": "{company}",
      "product_sold": "{product}",
      "opportunities": [
        {{
          "signal": "description",
          "business_implication": "possible implication",
          "relevance_score": 5,
          "evidence_status": "unverified",
          "evidence_gap": "missing evidence",
          "next_research_action": "action"
        }}
      ],
      "overall_assessment": "summary",
      "recommended_next_step": "action"
    }}
    """

    return ask_agent_json(prompt)


# =====================================================
# 6. AGENT 3 — EVIDENCE AUDITOR
# =====================================================

def evidence_auditor(company, analyst, sources):
    prompt = f"""
    You are Agent 3: Evidence Auditor.

    Company: {company}

    Agent 2 analysis:
    {json.dumps(analyst, ensure_ascii=False)}

    Retrieved public news records:
    {json.dumps(sources, ensure_ascii=False)}

    Audit each proposed opportunity.

    Allowed evidence_status values:
    supported, partially_supported, unsupported,
    no_relevant_source.

    For every opportunity return:
    signal, evidence_status, supporting_sources,
    evidence_explanation, unsupported_assumptions,
    follow_up_needed.

    supporting_sources must be a list of URLs copied
    exactly from the supplied source records.

    Rules:
    - Use only the supplied source records.
    - Never invent URLs, headlines or publication dates.
    - A company mention does not automatically support a claim.
    - A headline is not proof of a company's buying intent.
    - Do not claim to have read full articles.
    - If no sources were supplied, use no_relevant_source.
    - Distinguish evidence about an event from a hypothesis
      about whether the event creates a need for the product.

    Return a JSON object:
    {{
      "company": "{company}",
      "audited_opportunities": [],
      "overall_evidence_quality": "low",
      "research_gaps": [],
      "recommended_follow_up": ""
    }}
    """

    result = ask_agent_json(prompt)

    if not isinstance(result, dict):
        raise ValueError(
            "Evidence Auditor did not return a JSON object."
        )

    return result


# =====================================================
# 7. AGENT 4 — SALES STRATEGIST
# =====================================================

def sales_strategist(company, industry, product, audit):
    prompt = f"""
    You are Agent 4: Sales Strategist.

    Company: {company}
    Industry: {industry}
    Product sold: {product}

    Audited research:
    {json.dumps(audit, ensure_ascii=False)}

    Produce an actionable sales plan.

    Return a JSON object with:
    company, opportunity_priority, rationale,
    discovery_questions, next_action,
    outreach_draft, evidence_caveat.

    Rules:
    - Prioritize evidence-backed opportunities.
    - If evidence is weak, recommend further research.
    - Do not invent company facts or contact details.
    - Do not imply that purchase intent is confirmed.
    - Make outreach relevant to the product and evidence.
    - Do not send messages or contact anyone.
    """

    return ask_agent_json(prompt)


# =====================================================
# 8. EVIDENCE GAP COUNTING
# =====================================================

def count_evidence_gaps(audit):
    """
    Count gap entries rather than characters in a string.
    """

    gaps = audit.get("research_gaps", [])

    if isinstance(gaps, str):
        return 1 if gaps.strip() else 0

    if isinstance(gaps, list):
        return len(gaps)

    return 0


def get_sources_from_audit(audit):
    """Collect cited source URLs without duplicates."""

    urls = []

    for opportunity in audit.get(
        "audited_opportunities", []
    ):
        sources = opportunity.get("supporting_sources", [])

        if isinstance(sources, str):
            sources = [sources]

        if isinstance(sources, list):
            for url in sources:
                if isinstance(url, str) and url not in urls:
                    urls.append(url)

    return urls


# =====================================================
# 9. WORKFLOW ORCHESTRATOR
# =====================================================

def run_signaldesk(company, industry, product):
    report = {
        "company": company,
        "industry": industry,
        "product_sold": product,
        "started_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "running",
        "steps": {},
        "errors": []
    }

    try:
        with st.status(
            "Running SignalDesk AI agents...",
            expanded=True
        ) as status:

            st.write("Agent 1: Signal Hunter")
            signals = signal_hunter(
                company, industry, product
            )
            report["steps"]["signal_hunter"] = signals

            st.write("Agent 2: Company Analyst")
            analyst = company_analyst(
                company, industry, product, signals
            )
            report["steps"]["company_analyst"] = analyst

            st.write("Retrieving public news...")
            try:
                sources = fetch_public_evidence(
                    company, industry
                )
                report["steps"]["public_sources"] = sources

            except Exception as exc:
                sources = []
                report["steps"]["public_sources"] = []
                report["errors"].append(
                    "News retrieval: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                )

            st.write("Agent 3: Evidence Auditor")
            audit = evidence_auditor(
                company, analyst, sources
            )
            report["steps"]["evidence_auditor"] = audit

            st.write("Agent 4: Sales Strategist")
            sales = sales_strategist(
                company, industry, product, audit
            )
            report["steps"]["sales_strategist"] = sales

            report["status"] = "completed"
            status.update(
                label="Research workflow completed",
                state="complete"
            )

    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(
            type(exc).__name__ + ": " + str(exc)
        )

    report["finished_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    return report


# =====================================================
# 10. DASHBOARD INPUT
# =====================================================

with st.sidebar:
    st.header("Research settings")
    st.caption("Analyze a target account.")
    st.caption(f"AI model: {MODEL}")

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


# =====================================================
# 11. RUN AND DISPLAY REPORT
# =====================================================

if submitted:
    if not company.strip() or not industry.strip() or not product.strip():
        st.warning("Please complete all three fields.")

    else:
        report = run_signaldesk(
            company.strip(),
            industry.strip(),
            product.strip()
        )

        st.session_state["latest_report"] = report


if "latest_report" in st.session_state:
    report = st.session_state["latest_report"]

    if report["status"] == "failed":
        st.error("The workflow did not complete.")

        for error in report["errors"]:
            st.write(error)

    else:
        steps = report["steps"]

        analyst = steps["company_analyst"]
        audit = steps["evidence_auditor"]
        sales = steps["sales_strategist"]
        sources = steps["public_sources"]

        st.success("Research report generated")

        if report["errors"]:
            with st.expander("Research warnings"):
                for error in report["errors"]:
                    st.warning(error)

        opportunities = analyst.get("opportunities", [])
        gaps_count = count_evidence_gaps(audit)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Potential opportunities",
            len(opportunities)
        )

        col2.metric(
            "Public news records",
            len(sources)
        )

        col3.metric(
            "Evidence gaps",
            gaps_count
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "Sales Strategy",
            "Opportunities",
            "Evidence",
            "Execution Log"
        ])

        # SALES STRATEGY
        with tab1:
            st.subheader("Recommended action")
            st.write(sales.get("next_action", "Not provided"))

            st.subheader("Opportunity rationale")
            st.write(sales.get("rationale", "Not provided"))

            st.subheader("Discovery questions")
            questions = sales.get("discovery_questions", [])

            if isinstance(questions, list):
                for question in questions:
                    st.markdown(f"- {question}")
            else:
                st.write(questions)

            st.subheader("Outreach draft")
            st.write(sales.get("outreach_draft", "Not provided"))

            st.caption(
                sales.get("evidence_caveat", "")
            )

        # OPPORTUNITIES
        with tab2:
            for index, opportunity in enumerate(
                opportunities, start=1
            ):
                with st.expander(
                    f"{index}. {opportunity.get('signal', 'Opportunity')}"
                ):
                    st.write(
                        "**Business implication:**",
                        opportunity.get(
                            "business_implication", ""
                        )
                    )

                    st.write(
                        "**Relevance score:**",
                        opportunity.get("relevance_score", "N/A"),
                        "/ 10"
                    )

                    st.write(
                        "**Evidence gap:**",
                        opportunity.get("evidence_gap", "")
                    )

                    st.write(
                        "**Next research action:**",
                        opportunity.get(
                            "next_research_action", ""
                        )
                    )

        # EVIDENCE
        with tab3:
            st.subheader("Evidence quality")
            st.write(
                audit.get(
                    "overall_evidence_quality", "Not assessed"
                )
            )

            st.subheader("Research gaps")
            gaps = audit.get("research_gaps", [])

            if isinstance(gaps, str):
                st.write(gaps)
            elif gaps:
                for gap in gaps:
                    st.markdown(f"- {gap}")
            else:
                st.write("No research gaps reported.")

            st.subheader("Audited opportunities")

            for item in audit.get(
                "audited_opportunities", []
            ):
                st.markdown(
                    f"**{item.get('signal', 'Signal')}**"
                )

                st.write(
                    "Status:",
                    item.get(
                        "evidence_status", "unassessed"
                    )
                )

                st.write(
                    item.get("evidence_explanation", "")
                )

                for url in item.get(
                    "supporting_sources", []
                ):
                    if isinstance(url, str):
                        st.markdown(f"- [Open cited source]({url})")

                st.divider()

            st.subheader("Retrieved public news")

            if sources:
                for source in sources:
                    st.markdown(
                        f"- [{source['title']}]({source['url']})"
                    )
                    st.caption(
                        f"{source['publisher']} | "
                        f"{source['published']}"
                    )
            else:
                st.warning(
                    "No public news records were retrieved. "
                    "The opportunities remain hypotheses; "
                    "do not treat them as verified buying signals."
                )

        # EXECUTION LOG
        with tab4:
            st.write("Workflow status:", report["status"])
            st.write("Started:", report["started_at"])
            st.write("Finished:", report.get("finished_at"))
            st.write("Warnings and errors:", report["errors"])

            st.write(
                "Completed stages:",
                list(steps.keys())
            )

            with st.expander("View complete report JSON"):
                st.json(report)

        st.download_button(
            "Download full research report",
            data=json.dumps(
                report, indent=2, ensure_ascii=False
            ),
            file_name="signaldesk_report.json",
            mime="application/json"
        )
