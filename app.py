

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
