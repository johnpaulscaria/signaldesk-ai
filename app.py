
import json
import os
import xml.etree.ElementTree as ET
from urllib.parse import quote
from datetime import datetime, timezone

import requests
import streamlit as st
from google import genai


st.set_page_config(
    page_title="SignalDesk AI",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 SignalDesk AI")
st.caption(
    "Multi-agent B2B sales intelligence | Research, "
    "evidence screening and sales strategy"
)

# Configure the API key through Streamlit Secrets.
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY", "")

if not api_key:
    st.error(
        "API key not configured. Add GEMINI_API_KEY "
        "to Streamlit app secrets."
    )
    st.stop()

client = genai.Client(api_key=api_key)
MODEL = "gemini-3.8-flash"


def ask_agent(prompt):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    if not response.text:
        raise ValueError("The AI model returned no text.")
    return response.text


def parse_json(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("Agent output was not valid JSON.")


def signal_hunter(company, industry, product):
    return ask_agent(f"""
    You are Signal Hunter, Agent 1.
    Company: {company}
    Industry: {industry}
    Product sold: {product}

    Identify five possible buying signals to investigate.
    For each include signal, why_it_matters,
    evidence_needed, and confidence (low, medium, high).

    Do not invent company news or facts.
    Return a JSON array only. Treat signals as hypotheses.
    """)


def company_analyst(company, industry, product, signals):
    return parse_json(ask_agent(f"""
    You are Company Analyst, Agent 2.
    Company: {company}
    Industry: {industry}
    Product: {product}

    Potential signals from Agent 1:
    {signals}

    Analyze commercial relevance. Return a JSON object:
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
          "next_research_action": "research action"
        }}
      ],
      "overall_assessment": "summary",
      "recommended_next_step": "action"
    }}

    Scores indicate relevance, not purchase probability.
    Do not invent verified facts.
    """))


def fetch_public_evidence(company, industry):
    query = quote(f'"{company}" {industry}')
    url = (
        "https://news.google.com/rss/search?q="
        + query + "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    response = requests.get(
        url,
        headers={"User-Agent": "SignalDeskAI/1.0"},
        timeout=20
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    results = []

    for item in root.findall(".//item")[:10]:
        source = item.find("source")
        results.append({
            "title": item.findtext("title", ""),
            "url": item.findtext("link", ""),
            "published": item.findtext("pubDate", ""),
            "publisher": (
                source.text if source is not None else ""
            )
        })

    return results


def evidence_auditor(company, analyst, sources):
    return parse_json(ask_agent(f"""
    You are Evidence Auditor, Agent 3.
    Company: {company}

    Agent 2 analysis:
    {json.dumps(analyst, ensure_ascii=False)}

    Public source records:
    {json.dumps(sources, ensure_ascii=False)}

    Return JSON with:
    company, audited_opportunities,
    overall_evidence_quality, research_gaps,
    recommended_follow_up.

    For each opportunity include:
    signal, evidence_status, supporting_sources,
    evidence_explanation, unsupported_assumptions,
    follow_up_needed.

    Allowed evidence statuses:
    supported, partially_supported, unsupported,
    no_relevant_source.

    Use only the supplied sources. Never invent URLs.
    Headlines are not full articles. Explain limitations.
    Do not infer confirmed purchase intent.
    """))


def sales_strategist(company, industry, product, audit):
    return parse_json(ask_agent(f"""
    You are Sales Strategist, Agent 4.
    Company: {company}
    Industry: {industry}
    Product: {product}

    Audited findings:
    {json.dumps(audit, ensure_ascii=False)}

    Return JSON with:
    company, opportunity_priority, rationale,
    discovery_questions, next_action, outreach_draft,
    evidence_caveat.

    Prioritize evidence-backed opportunities.
    If evidence is weak, recommend further research.
    Do not invent contacts or company facts.
    Do not send messages.
    """))


def run_workflow(company, industry, product):
    report = {
        "company": company,
        "industry": industry,
        "product_sold": product,
        "started_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "steps": {},
        "errors": []
    }

    with st.status(
        "Running the sales intelligence team...",
        expanded=True
    ) as status:
        try:
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

            st.write("Retrieving public news evidence")
            try:
                sources = fetch_public_evidence(
                    company, industry
                )
            except Exception as exc:
                sources = []
                report["errors"].append(
                    "News retrieval failed: " + str(exc)
                )

            report["steps"]["public_sources"] = sources

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

            report["finished_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            status.update(
                label="Research workflow completed",
                state="complete"
            )

        except Exception as exc:
            report["errors"].append(
                type(exc).__name__ + ": " + str(exc)
            )
            status.update(
                label="Workflow failed",
                state="error"
            )

    return report


with st.sidebar:
    st.header("Research settings")
    st.caption("Enter the business you want to investigate.")

with st.form("research_form"):
    company = st.text_input(
        "Target company",
        placeholder="e.g. Tata Motors"
    )
    industry = st.text_input(
        "Industry",
        value="Automotive manufacturing"
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
    else:
        with st.spinner("Researching your target..."):
            report = run_workflow(
                company.strip(),
                industry.strip(),
                product.strip()
            )

        if report["errors"] and not report["steps"].get(
            "sales_strategist"
        ):
            st.error("The workflow encountered an error.")
            for error in report["errors"]:
                st.write(error)

        if report["steps"].get("sales_strategist"):
            analyst = report["steps"]["company_analyst"]
            audit = report["steps"]["evidence_auditor"]
            sales = report["steps"]["sales_strategist"]
            sources = report["steps"]["public_sources"]

            st.success("Research report generated")

            a, b, c = st.columns(3)
            a.metric(
                "Potential opportunities",
                len(analyst.get("opportunities", []))
            )
            b.metric("Public news records", len(sources))
            c.metric(
                "Evidence gaps",
                len(audit.get("research_gaps", []))
            )

            tab1, tab2, tab3, tab4 = st.tabs([
                "Sales Strategy",
                "Opportunities",
                "Evidence",
                "Execution Log"
            ])

            with tab1:
                st.subheader("Recommended action")
                st.write(sales.get("next_action", ""))
                st.subheader("Opportunity rationale")
                st.write(sales.get("rationale", ""))
                st.subheader("Discovery questions")
                st.write(
                    sales.get("discovery_questions", [])
                )
                st.subheader("Outreach draft")
                st.write(sales.get("outreach_draft", ""))
                st.caption(
                    sales.get("evidence_caveat", "")
                )

            with tab2:
                st.json(analyst.get("opportunities", []))

            with tab3:
                st.write("Evidence quality")
                st.write(
                    audit.get("overall_evidence_quality", "")
                )
                st.write("Research gaps")
                st.write(audit.get("research_gaps", []))
                st.write("Audited opportunities")
                st.json(
                    audit.get("audited_opportunities", [])
                )
                st.subheader("Retrieved public sources")
                for source in sources:
                    st.markdown(
                        f"- [{source['title']}]({source['url']})"
                        f" — {source['publisher']} "
                        f"({source['published']})"
                    )

            with tab4:
                st.json({
                    "started_at": report["started_at"],
                    "finished_at": report.get("finished_at"),
                    "errors": report["errors"],
                    "completed_agents": list(
                        report["steps"].keys()
                    )
                })

            st.download_button(
                "Download full JSON report",
                data=json.dumps(
                    report, indent=2, ensure_ascii=False
                ),
                file_name="signaldesk_report.json",
                mime="application/json"
            )
