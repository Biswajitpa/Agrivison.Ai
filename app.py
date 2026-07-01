# ╔══════════════════════════════════════════════════════════════════════════╗
# ║              AGRIVION AI  –  Production App  v2.0                       ║
# ║   Single-file Streamlit application with all features integrated         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os, io, json, time, base64, hashlib, random, math, warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw, ImageFilter
import requests
import streamlit as st
# Move this OUTSIDE the page block, at the top of your file with other imports
from fpdf import FPDF
from datetime import datetime
import numpy as np
import pandas as pd
# Move this OUTSIDE the page block, at the top of your file with other imports
from fpdf import FPDF
from datetime import datetime
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

# ── Environment ────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
WEATHER_API_KEY   = os.getenv("OPENWEATHER_API_KEY", "")
MODEL_PATH        = os.getenv("MODEL_PATH",       "models/AgriVisionAI_Final.h5")
CROP_MODEL_PATH   = os.getenv("CROP_MODEL_PATH",  "models/crop_model.pkl")
SCALER_PATH       = os.getenv("SCALER_PATH",      "models/scaler.pkl")
LABEL_ENCODER_PATH= os.getenv("LABEL_ENCODER_PATH","models/label_encoder.pkl")

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agrivion AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GLOBAL CSS  –  Production Design System                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@400,0..1&display=swap');

/* ── Design Tokens ── */
:root {
  --primary:          #006e2f;
  --primary-light:    #22c55e;
  --primary-dim:      #4ae176;
  --primary-bg:       rgba(0,110,47,.08);
  --primary-border:   rgba(0,110,47,.2);
  --secondary:        #1f6c3a;
  --surface:          #f7f9fb;
  --surface-card:     #ffffff;
  --surface-low:      #f2f4f6;
  --surface-high:     #e6e8ea;
  --border:           #E2E8F0;
  --border-soft:      #bccbb9;
  --text-primary:     #191c1e;
  --text-secondary:   #3d4a3d;
  --text-muted:       #6d7b6c;
  --error:            #ba1a1a;
  --error-bg:         #ffdad6;
  --warn:             #f97316;
  --warn-bg:          #fff7ed;
  --info:             #1d4ed8;
  --info-bg:          #dbeafe;
  --success-bg:       #dcfce7;
  --shadow-sm:   0 2px 8px rgba(15,23,42,.06);
  --shadow-md:   0 4px 20px rgba(15,23,42,.08);
  --shadow-lg:   0 10px 40px rgba(15,23,42,.12);
  --shadow-green:0 8px 24px rgba(0,110,47,.25);
  --radius-sm:   8px;
  --radius-md:   12px;
  --radius-lg:   16px;
  --radius-xl:   20px;
  --radius-full: 9999px;
  --transition:  all 0.25s cubic-bezier(0.4,0,0.2,1);
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
div[data-testid="collapsedControl"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1f12 0%, #0d2b18 50%, #0a1f12 100%) !important;
    border-right: 1px solid rgba(34,197,94,.12) !important;
    width: 270px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebarNav"] { display: none !important; }
section[data-testid="stSidebar"] > div { background: transparent !important; }

/* ── Main area ── */
.main .block-container {
    padding: 1.75rem 2rem !important;
    max-width: 1440px !important;
}

/* ──────────────── CARD SYSTEM ──────────────── */
.agri-card {
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}
.agri-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(0,110,47,.3), transparent);
    opacity: 0;
    transition: var(--transition);
}
.agri-card:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
.agri-card:hover::before { opacity: 1; }

.card-glass {
    background: rgba(255,255,255,.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,.4);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow-sm);
}
.card-gradient-green {
    background: linear-gradient(135deg, #006e2f 0%, #1f6c3a 50%, #145228 100%);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-green);
}
.card-gradient-blue {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    color: white;
}
.card-gradient-weather {
    background: linear-gradient(135deg, rgba(0,110,47,.06) 0%, rgba(29,78,216,.06) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
}

/* ──────────────── KPI CARDS ──────────────── */
.kpi-grid { display: grid; grid-template-columns: repeat(6,1fr); gap: 1rem; margin: 1.25rem 0; }
.kpi-card {
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 var(--radius-md) var(--radius-md);
}
.kpi-card.green::after  { background: linear-gradient(90deg, #006e2f, #22c55e); }
.kpi-card.blue::after   { background: linear-gradient(90deg, #1d4ed8, #60a5fa); }
.kpi-card.orange::after { background: linear-gradient(90deg, #f97316, #fb923c); }
.kpi-card.purple::after { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
.kpi-card.teal::after   { background: linear-gradient(90deg, #0d9488, #2dd4bf); }
.kpi-card.red::after    { background: linear-gradient(90deg, #ba1a1a, #f87171); }
.kpi-icon {
    width: 42px; height: 42px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; margin-bottom: .875rem;
}
.kpi-label { font-size: 11px; color: var(--text-muted); font-weight: 600;
             text-transform: uppercase; letter-spacing: .06em; }
.kpi-value { font-size: 26px; font-weight: 800; color: var(--text-primary);
             line-height: 1.1; margin-top: .25rem; }
.kpi-delta { font-size: 11px; font-weight: 700; margin-top: .25rem; }
.kpi-delta.up   { color: #006e2f; }
.kpi-delta.down { color: #ba1a1a; }

/* ──────────────── BADGES ──────────────── */
.badge { border-radius: var(--radius-full); padding: 3px 12px;
         font-size: 11px; font-weight: 700; display: inline-block; }
.badge-green  { background: var(--success-bg); color: #166534; }
.badge-red    { background: var(--error-bg);   color: #93000a; }
.badge-orange { background: var(--warn-bg);    color: #9a3412; }
.badge-blue   { background: var(--info-bg);    color: #1e3a8a; }
.badge-purple { background: #f3e8ff; color: #6b21a8; }
.badge-live   { background: var(--primary-bg); color: var(--primary);
                display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; }

/* ──────────────── ALERTS ──────────────── */
.alert { border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
         padding: 1rem 1.125rem; margin-bottom: .625rem;
         border-left-width: 4px; border-left-style: solid; }
.alert-danger  { border-color: var(--error);  background: rgba(186,26,26,.06); }
.alert-warning { border-color: var(--warn);   background: rgba(249,115,22,.06); }
.alert-success { border-color: var(--primary);background: rgba(0,110,47,.06); }
.alert-info    { border-color: var(--info);   background: rgba(29,78,216,.06); }

/* ──────────────── AI INSIGHT BOX ──────────────── */
.ai-box {
    background: linear-gradient(135deg, rgba(0,110,47,.05) 0%, rgba(74,225,118,.03) 100%);
    border: 1px solid rgba(0,110,47,.18);
    border-radius: var(--radius-md);
    padding: 1.125rem;
    display: flex; gap: 1rem; align-items: flex-start;
}
.ai-orb {
    width: 40px; height: 40px; flex-shrink: 0;
    background: linear-gradient(135deg, var(--primary), var(--primary-light));
    border-radius: var(--radius-full);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; box-shadow: 0 4px 12px rgba(0,110,47,.3);
}

/* ──────────────── SENSOR CHIP ──────────────── */
.sensor-chip {
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: .875rem;
    transition: var(--transition);
}
.sensor-chip:hover { border-color: var(--primary-light); box-shadow: var(--shadow-sm); }
.sensor-online { width: 8px; height: 8px; background: #22c55e;
                 border-radius: 50%; display: inline-block;
                 box-shadow: 0 0 0 3px rgba(34,197,94,.2); }
.sensor-offline { width: 8px; height: 8px; background: #ba1a1a;
                  border-radius: 50%; display: inline-block; }

/* ──────────────── PROGRESS ──────────────── */
.prog-track { background: var(--surface-high); border-radius: var(--radius-full); height: 7px; overflow: hidden; }
.prog-bar { height: 7px; border-radius: var(--radius-full);
            background: linear-gradient(90deg, var(--primary), var(--primary-light));
            transition: width .8s ease; }
.prog-bar.blue   { background: linear-gradient(90deg, #1d4ed8, #60a5fa); }
.prog-bar.orange { background: linear-gradient(90deg, #f97316, #fb923c); }
.prog-bar.red    { background: linear-gradient(90deg, #ba1a1a, #f87171); }

/* ──────────────── PAGE HEADER ──────────────── */
.page-hero {
    background: linear-gradient(135deg, rgba(0,110,47,.05) 0%, rgba(34,197,94,.03) 100%);
    border: 1px solid rgba(0,110,47,.1);
    border-radius: var(--radius-xl);
    padding: 1.75rem 2rem;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
}
.page-hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(34,197,94,.08) 0%, transparent 70%);
    border-radius: 50%;
}
.page-title   { font-size: 30px; font-weight: 800; color: var(--text-primary);
                letter-spacing: -.03em; line-height: 1.15; }
.page-subtitle{ font-size: 15px; color: var(--text-secondary); margin-top: .25rem; }

/* ──────────────── WEATHER CARD ──────────────── */
.weather-hero {
    background: linear-gradient(135deg, #0a1f12 0%, #0d3b21 50%, #0a2e1b 100%);
    border-radius: var(--radius-xl);
    padding: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.weather-hero::after {
    content: '';
    position: absolute;
    top: -30%; right: -10%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(34,197,94,.12) 0%, transparent 70%);
    border-radius: 50%;
}
.temp-big { font-size: 72px; font-weight: 900; line-height: 1; letter-spacing: -.04em; }
.forecast-pill {
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: var(--radius-md);
    padding: .875rem .625rem;
    text-align: center;
    transition: var(--transition);
    flex: 1;
}
.forecast-pill:hover { background: rgba(255,255,255,.15); }
.forecast-pill.today { background: rgba(34,197,94,.2); border-color: rgba(34,197,94,.4); }

/* ──────────────── CHAT UI ──────────────── */
.chat-container {
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    height: 420px;
    overflow-y: auto;
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    gap: .75rem;
    scroll-behavior: smooth;
}
.chat-container::-webkit-scrollbar { width: 5px; }
.chat-container::-webkit-scrollbar-track { background: transparent; }
.chat-container::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
.msg-user {
    align-self: flex-end;
    background: linear-gradient(135deg, var(--primary), #1a7d3a);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: .75rem 1rem;
    max-width: 78%;
    font-size: 14px;
    box-shadow: 0 4px 12px rgba(0,110,47,.25);
}
.msg-bot {
    align-self: flex-start;
    background: var(--surface-low);
    border: 1px solid var(--border);
    color: var(--text-primary);
    border-radius: 18px 18px 18px 4px;
    padding: .75rem 1rem;
    max-width: 78%;
    font-size: 14px;
}
.msg-bot strong { color: var(--primary); }
.msg-time { font-size: 10px; opacity: .5; margin-top: .25rem; text-align: right; }
.typing-dot {
    width: 8px; height: 8px;
    background: var(--text-muted);
    border-radius: 50%;
    display: inline-block;
    animation: typing 1.4s infinite;
}
@keyframes typing {
    0%,60%,100% { transform: translateY(0); opacity: .4; }
    30%           { transform: translateY(-6px); opacity: 1; }
}

/* ──────────────── NUTRIENT GAUGE ──────────────── */
.nutrient-bar-wrap {
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem;
    margin-bottom: .5rem;
}

/* ──────────────── MAP MOCK ──────────────── */
.farm-map-container {
    background: linear-gradient(145deg, #1a3d2b 0%, #143020 40%, #0d2217 100%);
    border-radius: var(--radius-lg);
    height: 380px;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(34,197,94,.15);
}
.map-grid {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(34,197,94,.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,.08) 1px, transparent 1px);
    background-size: 40px 40px;
}
.map-pin {
    position: absolute;
    width: 12px; height: 12px;
    background: #22c55e;
    border-radius: 50%;
    border: 2px solid white;
    box-shadow: 0 0 0 4px rgba(34,197,94,.3), 0 0 20px rgba(34,197,94,.5);
    animation: ping 2s cubic-bezier(0,0,.2,1) infinite;
}
@keyframes ping {
    75%,100% { box-shadow: 0 0 0 12px rgba(34,197,94,0), 0 0 20px rgba(34,197,94,0); }
}
.map-label {
    position: absolute;
    background: rgba(0,0,0,.7);
    color: white;
    border: 1px solid rgba(34,197,94,.4);
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 600;
    backdrop-filter: blur(8px);
}
.map-overlay {
    position: absolute;
    bottom: 1rem; left: 1rem;
    background: rgba(0,0,0,.6);
    border: 1px solid rgba(34,197,94,.3);
    border-radius: var(--radius-md);
    padding: .875rem 1rem;
    color: white;
    backdrop-filter: blur(12px);
    font-size: 13px;
}
.map-controls {
    position: absolute;
    top: 1rem; right: 1rem;
    display: flex; flex-direction: column; gap: 4px;
}
.map-ctrl-btn {
    width: 32px; height: 32px;
    background: rgba(0,0,0,.6);
    border: 1px solid rgba(255,255,255,.15);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 16px; cursor: pointer;
    backdrop-filter: blur(8px);
}

/* ──────────────── ANIMATIONS ──────────────── */
@keyframes fadeInUp {
    from { opacity:0; transform: translateY(16px); }
    to   { opacity:1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity:0; } to { opacity:1; }
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position:  200% 0; }
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 0 0   rgba(34,197,94,.4); }
    50%      { box-shadow: 0 0 0 8px rgba(34,197,94,0);  }
}
.animate-fadein   { animation: fadeInUp .5s ease both; }
.animate-fadein-1 { animation: fadeInUp .5s .1s ease both; }
.animate-fadein-2 { animation: fadeInUp .5s .2s ease both; }
.animate-fadein-3 { animation: fadeInUp .5s .3s ease both; }
.live-pulse { animation: pulse-green 2s infinite; }
.shimmer-bg {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 6px;
}

/* ──────────────── SIDEBAR NAV ──────────────── */
.nav-logo-wrap { padding: 1.5rem 1.25rem 1rem; border-bottom: 1px solid rgba(255,255,255,.06); }
.nav-logo { font-size: 20px; font-weight: 900; color: #4ae176; letter-spacing: -.02em; }
.nav-sub { font-size: 9px; color: rgba(255,255,255,.35); text-transform: uppercase;
           letter-spacing: .12em; margin-top: 2px; }
.nav-section { font-size: 9px; font-weight: 700; letter-spacing: .12em;
               text-transform: uppercase; color: rgba(255,255,255,.25);
               padding: 1.25rem 1.25rem .375rem; }
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: .65rem 1.25rem;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    font-size: 13.5px; font-weight: 500; color: rgba(255,255,255,.55);
    cursor: pointer; transition: var(--transition);
    border-left: 3px solid transparent;
    margin-right: .5rem;
}
.nav-item:hover { background: rgba(255,255,255,.06); color: rgba(255,255,255,.85); }
.nav-item.active {
    background: rgba(34,197,94,.12);
    border-color: #22c55e;
    color: #4ae176; font-weight: 700;
}
.nav-item .nav-icon { width: 18px; text-align: center; font-size: 15px; }
.nav-user {
    padding: 1rem 1.25rem;
    border-top: 1px solid rgba(255,255,255,.06);
    display: flex; align-items: center; gap: .875rem;
}
.nav-avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #006e2f, #22c55e);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 800; color: white;
    flex-shrink: 0;
    box-shadow: 0 0 0 2px rgba(34,197,94,.4);
}

/* ──────────────── TABLE ──────────────── */
.ag-table { width:100%; border-collapse:collapse; font-size:13.5px; font-family:'Inter',sans-serif; }
.ag-table th {
    background: var(--surface-low);
    color: var(--text-muted);
    padding: .75rem 1rem;
    text-align:left;
    font-size: 10.5px; font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
}
.ag-table td { padding:.8rem 1rem; border-bottom:1px solid rgba(226,232,240,.5); color:var(--text-primary); }
.ag-table tr:last-child td { border-bottom: none; }
.ag-table tr:hover td { background: rgba(0,110,47,.02); }

/* ──────────────── WIDGET OVERRIDES ──────────────── */
div[data-testid="stMetric"] {
    background: var(--surface-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.125rem !important;
    box-shadow: var(--shadow-sm) !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #006e2f, #1a7d3a) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13.5px !important;
    letter-spacing: .01em !important;
    transition: var(--transition) !important;
    box-shadow: 0 2px 8px rgba(0,110,47,.25) !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #005321, #006e2f) !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-green) !important;
}
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextInput"] > div > div > input,
div[data-testid="stNumberInput"] > div > div > input,
textarea {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    background: var(--surface-card) !important;
}
div[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 13.5px !important;
    color: var(--text-muted) !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
}
div[data-testid="stFileUploader"] {
    border-radius: var(--radius-md) !important;
    border-color: var(--border) !important;
}
[data-testid="stSlider"] .stSlider > div > div > div {
    background: var(--primary) !important;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--primary-light)) !important;
    border-radius: var(--radius-full) !important;
}

/* ──────────────── SCROLLBAR ──────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-soft); }

/* ──────────────── SECTION TITLE ──────────────── */
.section-title {
    font-size: 18px; font-weight: 700; color: var(--text-primary);
    margin-bottom: 1rem; letter-spacing: -.01em;
    display: flex; align-items: center; gap: .5rem;
}
.section-title::after {
    content:''; flex:1; height:1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ──────────────── FOOTER ──────────────── */
.ag-footer {
    border-top: 1px solid var(--border);
    padding: 1.5rem 0 .5rem;
    margin-top: 2.5rem;
    display: flex; justify-content: space-between; align-items: center;
    font-size: 12.5px; color: var(--text-muted);
}
.ag-footer a { color: var(--text-muted); text-decoration: none; transition: var(--transition); }
.ag-footer a:hover { color: var(--primary); }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HELPER UTILITIES                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner=False)
def load_disease_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        with open("models/disease_labels.json") as f:
            labels = json.load(f)
        return model, labels
    except Exception:
        return None, [
            "Apple Scab","Black Rot","Cedar Apple Rust","Healthy Apple",
            "Bacterial Spot","Healthy Peach","Brown Rust",
            "Powdery Mildew","Nitrogen Deficiency","Leaf Blight",
            "Tomato Late Blight","Healthy Wheat"
        ]

@st.cache_resource(show_spinner=False)
def load_crop_model():
    try:
        import pickle, joblib
        with open(CROP_MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception:
            scaler = None
        return data["model"], data["crops"], scaler
    except Exception:
        return None, CROP_NAMES, None

CROP_NAMES = [
    "Rice","Maize","Chickpea","Kidney Beans","Pigeon Peas",
    "Moth Beans","Mung Bean","Black Gram","Lentil","Pomegranate",
    "Banana","Mango","Grapes","Watermelon","Muskmelon","Apple",
    "Orange","Papaya","Coconut","Cotton","Jute","Coffee","Wheat"
]

DISEASE_TREATMENTS = {
    "Brown Rust":         {"fungicide":"Tebuconazole 250g/ha + Azoxystrobin 200ml/ha","env":"Reduce night irrigation, improve canopy ventilation","urgency":"High"},
    "Powdery Mildew":     {"fungicide":"Sulfur dust 3kg/ha or Triadimefon 0.5g/L","env":"Improve air circulation, reduce humidity below 60%","urgency":"Medium"},
    "Apple Scab":         {"fungicide":"Captan 80WP 2g/L or Myclobutanil 0.5g/L","env":"Remove infected leaves, ensure good drainage","urgency":"Medium"},
    "Black Rot":          {"fungicide":"Mancozeb 2g/L spray every 10 days","env":"Remove mummified fruits, prune infected canes","urgency":"High"},
    "Leaf Blight":        {"fungicide":"Propiconazole 25EC @ 1ml/L","env":"Avoid overhead irrigation, maintain field hygiene","urgency":"High"},
    "Bacterial Spot":     {"fungicide":"Copper oxychloride 3g/L","env":"Use disease-free seeds, crop rotation","urgency":"Medium"},
    "Nitrogen Deficiency":{"fungicide":"Apply urea 46% N @ 30kg/acre","env":"Soil pH correction to 6.0-6.5 for better uptake","urgency":"Low"},
    "Tomato Late Blight": {"fungicide":"Metalaxyl 8% + Mancozeb 64% @ 2.5g/L","env":"Remove infected plants immediately, avoid wetting foliage","urgency":"Critical"},
}

def get_weather(city="Bhubaneswar"):
    """Fetch weather or return realistic mock data."""
    if WEATHER_API_KEY and WEATHER_API_KEY != "your_openweather_api_key_here":
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
            r = requests.get(url, timeout=4)
            d = r.json()
            return {
                "temp": round(d["main"]["temp"]),
                "feels": round(d["main"]["feels_like"]),
                "humidity": d["main"]["humidity"],
                "wind": round(d["wind"]["speed"] * 3.6),
                "pressure": d["main"]["pressure"],
                "desc": d["weather"][0]["description"].title(),
                "city": city, "uv": 4,
            }
        except Exception:
            pass
    # Realistic mock
    return {
        "temp": 28, "feels": 30, "humidity": 65,
        "wind": 14, "pressure": 1012,
        "desc": "Partly Cloudy", "city": city, "uv": 5,
    }

FORECAST_MOCK = [
    ("Mon","☀️", 30, 20, 0.0),
    ("Tue","☀️", 32, 21, 0.0),
    ("Wed","⛅", 27, 19, 2.1),
    ("Thu","🌧️",23, 17,11.4),
    ("Fri","🌦️",25, 18, 3.2),
    ("Sat","☀️", 29, 19, 0.0),
    ("Sun","☀️", 31, 20, 0.0),
]

def groq_chat(messages, lang="English"):
    """Call Groq LLM API."""
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        return _fallback_response(messages[-1]["content"] if messages else "", lang)
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        sys_prompt = (
            f"You are AgriBot, an expert AI agricultural assistant for Agrivion AI. "
            f"You help farmers with crop diseases, irrigation, yield, weather advisories, and farm management. "
            f"Respond in {lang}. Be concise, practical, and use simple language. "
            f"Use emojis and bullet points to make responses readable. "
            f"Always end with a helpful tip or next step."
        )
        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "system", "content": sys_prompt}] + messages,
            "temperature": 0.7, "max_tokens": 512,
        }
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers=headers, json=payload, timeout=10)
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return _fallback_response(messages[-1]["content"] if messages else "", lang)

def _fallback_response(query: str, lang: str) -> str:
    q = query.lower()
    if any(w in q for w in ["disease","rust","blight","mildew","fungus","pest"]):
        return ("🌿 **Disease Advisory**\n\nFor fungal diseases:\n"
                "• Apply **Tebuconazole** or **Azoxystrobin** fungicide\n"
                "• Maintain humidity below 65%\n"
                "• Ensure good canopy ventilation\n"
                "• Remove infected plant material promptly\n\n"
                "💡 *Tip: Early detection saves up to 40% of yield losses.*")
    if any(w in q for w in ["water","irrigat","moisture","dry"]):
        return ("💧 **Irrigation Advisory**\n\n"
                "• Optimal soil moisture: **40-60% VWC**\n"
                "• Water early morning (5–8 AM) to reduce evaporation\n"
                "• Use drip irrigation for 30% water savings\n"
                "• Check tensiometer readings before irrigating\n\n"
                "💡 *Tip: Mulching reduces water need by 25%.*")
    if any(w in q for w in ["yield","harvest","crop","ton"]):
        return ("🌾 **Yield Optimization Tips**\n\n"
                "• Balanced N:P:K (4:2:1 ratio) boosts yield by 15-20%\n"
                "• Timely pest control protects 10-30% of potential yield\n"
                "• Optimal planting density improves sunlight utilization\n"
                "• Harvest at correct moisture content (14% for grains)\n\n"
                "💡 *Tip: Soil testing every 2 seasons helps fine-tune fertilization.*")
    if any(w in q for w in ["weather","rain","temperature","humidity"]):
        return ("🌤️ **Weather Advisory**\n\n"
                "• Current forecast shows mild conditions — ideal for field operations\n"
                "• Rain expected Thursday — postpone fertilizer application\n"
                "• UV Index is High — avoid spraying between 11 AM–3 PM\n"
                "• Wind > 15 km/h: Do not spray pesticides\n\n"
                "💡 *Tip: Use micro-climate monitoring for precision decisions.*")
    return ("🤖 **AgriBot at your service!**\n\n"
            "I can help you with:\n"
            "• 🌿 Crop disease diagnosis & treatment\n"
            "• 💧 Smart irrigation scheduling\n"
            "• 🌾 Yield prediction & harvest planning\n"
            "• 🌤️ Weather-based farm advisories\n"
            "• 🧪 Soil nutrient recommendations\n\n"
            "Please ask me anything about your farm!")

def predict_disease_mock(img: Image.Image):
    seed = sum(img.tobytes()[:100]) % 1000
    diseases = list(DISEASE_TREATMENTS.keys()) + ["Healthy Wheat","Healthy Apple","Healthy Peach"]
    idx  = seed % len(diseases)
    conf = 0.72 + (seed % 28) / 100
    return diseases[idx], conf

def predict_crop_model(N, P, K, temp, humidity, ph, rainfall):
    model, crops, scaler = load_crop_model()
    if model:
        try:
            X = np.array([[N, P, K, temp, humidity, ph, rainfall]], dtype=float)
            if scaler:
                X = scaler.transform(X)
            pred = model.predict(X)[0]
            return crops[pred], 0.78 + random.uniform(0, 0.18)
        except Exception:
            pass
    # Deterministic mock
    if rainfall > 200 and temp > 22: return "Rice", 0.94
    if temp > 20 and humidity > 50:  return "Maize", 0.88
    if ph < 6.5 and temp < 22:       return "Wheat", 0.91
    return "Cotton", 0.82

def nutrient_status(N, P, K, ph):
    def grade(v, low, high):
        if v < low: return "Deficient", "#ba1a1a"
        if v > high: return "Excess", "#f97316"
        return "Optimal", "#006e2f"
    return {
        "Nitrogen (N)":   grade(N,  60, 140),
        "Phosphorus (P)": grade(P,  25,  75),
        "Potassium (K)":  grade(K,  30, 100),
        "pH Balance":     grade(ph,  6.0, 7.5),
    }

def kpi(icon, label, value, delta=None, color="green"):
    delta_html = ""
    if delta:
        arrow = "▲" if not delta.startswith("-") else "▼"
        cls   = "up" if not delta.startswith("-") else "down"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    return f"""
<div class="kpi-card {color} animate-fadein">
    <div class="kpi-icon" style="background:{'rgba(0,110,47,.1)' if color=='green' else
                                               'rgba(29,78,216,.1)' if color=='blue' else
                                               'rgba(249,115,22,.1)' if color=='orange' else
                                               'rgba(124,58,237,.1)' if color=='purple' else
                                               'rgba(13,148,136,.1)' if color=='teal' else
                                               'rgba(186,26,26,.1)'}">
        {icon}
    </div>
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
    {delta_html}
</div>"""

def progress_bar(val, color=""):
    return f"""
<div class="prog-track">
    <div class="prog-bar {color}" style="width:{min(val,100)}%"></div>
</div>"""

def badge(text, color="green"):
    return f'<span class="badge badge-{color}">{text}</span>'

def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

def footer():
    st.markdown("""
<div class="ag-footer">
    <div>🌱 <strong style="color:#006e2f">Agrivion AI</strong> &nbsp;·&nbsp; Smart Agriculture Platform &nbsp;·&nbsp; © 2026</div>
    <div style="display:flex;gap:1.5rem">
        <a href="#">Support</a>
        <a href="#">Documentation</a>
        <a href="#">Privacy Policy</a>
    </div>
</div>""", unsafe_allow_html=True)

def plotly_defaults():
    """Base Plotly layout — xaxis/yaxis intentionally excluded to avoid duplicate-kwarg errors."""
    return dict(
        plot_bgcolor="#fff",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family="Inter", color="#3d4a3d", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", bordercolor="#E2E8F0",
                        font=dict(family="Inter", size=12)),
    )

# Axis style helpers — call inside update_layout as xaxis=_xax(), yaxis=_yax()
def _xax(**kw):
    d = dict(showgrid=False, color="#6d7b6c", linecolor="#E2E8F0")
    d.update(kw); return d

def _yax(**kw):
    d = dict(showgrid=True, gridcolor="rgba(226,232,240,.5)",
             color="#6d7b6c", linecolor="#E2E8F0")
    d.update(kw); return d


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR NAVIGATION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PAGES = [
    ("📊", "Dashboard"),
    ("🌿", "Crop Monitoring"),
    ("🔬", "AI Disease Detection"),
    ("💧", "Smart Irrigation"),
    ("🌤️",  "Weather Analytics"),
    ("📈", "Yield Prediction"),
    ("📡", "IoT Sensors"),
    ("🚜", "Farm Management"),
    ("🗺️",  "Live Farm Map"),
    ("📋", "Reports & Analytics"),
    ("🔔", "Alert Center"),
    ("🤖", "AI Chat Assistant"),
    ("🌱", "Crop Recommendation"),
    ("⚙️",  "Settings"),
]

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content":
         "👋 Hello! I'm **AgriBot**, your AI farming assistant.\n\n"
         "Ask me about crop diseases, irrigation, yield, or weather!"}
    ]
if "chat_lang" not in st.session_state:
    st.session_state.chat_lang = "English"

with st.sidebar:

    # ── Sidebar CSS (scoped, no conflicts) ─────────────────────────────────
    st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#071a0d 0%,#0a2414 60%,#071a0d 100%) !important;
}

/* Hide default radio circle + label spacing */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    padding: 0.6rem 1.1rem !important;
    margin: 1px 8px 1px 0 !important;
    border-radius: 0 8px 8px 0 !important;
    border-left: 3px solid transparent !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    color: rgba(255,255,255,0.55) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.4 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.07) !important;
    color: rgba(255,255,255,0.9) !important;
}
/* Selected item */
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + div label,
[data-testid="stSidebar"] [data-testid="stRadio"] div:has(input:checked) label {
    background: rgba(34,197,94,0.13) !important;
    border-left-color: #22c55e !important;
    color: #4ae176 !important;
    font-weight: 700 !important;
}
/* Hide the actual radio dot */
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}
/* Section headers */
[data-testid="stSidebar"] [data-testid="stRadio"] > label:first-child {
    display: none !important;
}
/* Divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 0.5rem 0 !important;
}
/* Sidebar markdown text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div {
    font-family: 'Inter', sans-serif !important;
}
/* Override any green button style leaking into sidebar */
[data-testid="stSidebar"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: rgba(255,255,255,0.6) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 0.6rem 1.1rem !important;
    border-radius: 0 8px 8px 0 !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] button:hover {
    background: rgba(255,255,255,0.07) !important;
    color: rgba(255,255,255,0.9) !important;
    transform: none !important;
}
</style>
""", unsafe_allow_html=True)

    # ── Logo ───────────────────────────────────────────────────────────────
    st.markdown("""
<div style="padding:1.4rem 1.2rem 0.8rem;border-bottom:1px solid rgba(255,255,255,0.07)">
    <div style="font-size:21px;font-weight:900;color:#4ae176;
                font-family:'Inter',sans-serif;letter-spacing:-0.02em">
        🌱 Agrivion AI
    </div>
    <div style="font-size:9.5px;color:rgba(255,255,255,0.3);text-transform:uppercase;
                letter-spacing:0.12em;margin-top:3px;font-family:'Inter',sans-serif">
        Smart Agriculture Platform · v2.0
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Navigation using radio ─────────────────────────────────────────────
    # Build flat list with section separators as disabled items
    NAV_OPTIONS = [
        "📊  Dashboard",
        "🌿  Crop Monitoring",
        "─── AI Features ───",
        "🔬  AI Disease Detection",
        "💧  Smart Irrigation",
        "🌤️   Weather Analytics",
        "📈  Yield Prediction",
        "🌱  Crop Recommendation",
        "─── Farm Operations ───",
        "📡  IoT Sensors",
        "🚜  Farm Management",
        "🗺️   Live Farm Map",
        "📋  Reports & Analytics",
        "🔔  Alert Center",
        "─── Intelligence ───",
        "🤖  AI Chat Assistant",
        "─── System ───",
        "⚙️   Settings",
    ]

    # Map display label → page name
    LABEL_TO_PAGE = {
        "📊  Dashboard":            "Dashboard",
        "🌿  Crop Monitoring":       "Crop Monitoring",
        "🔬  AI Disease Detection":  "AI Disease Detection",
        "💧  Smart Irrigation":      "Smart Irrigation",
        "🌤️   Weather Analytics":    "Weather Analytics",
        "📈  Yield Prediction":      "Yield Prediction",
        "🌱  Crop Recommendation":   "Crop Recommendation",
        "📡  IoT Sensors":           "IoT Sensors",
        "🚜  Farm Management":       "Farm Management",
        "🗺️   Live Farm Map":         "Live Farm Map",
        "📋  Reports & Analytics":   "Reports & Analytics",
        "🔔  Alert Center":          "Alert Center",
        "🤖  AI Chat Assistant":     "AI Chat Assistant",
        "⚙️   Settings":             "Settings",
    }
    PAGE_TO_LABEL = {v: k for k, v in LABEL_TO_PAGE.items()}

    # Selectable options only (no separators)
    SELECTABLE = [o for o in NAV_OPTIONS if not o.startswith("───")]

    # Find current selection
    current_label = PAGE_TO_LABEL.get(st.session_state.page, "📊  Dashboard")

    # Render sections with markdown headers + individual buttons
    _section_map = {
        "📊  Dashboard":           None,
        "🌿  Crop Monitoring":     None,
        "🔬  AI Disease Detection":"🤖 AI Features",
        "💧  Smart Irrigation":    "🤖 AI Features",
        "🌤️   Weather Analytics":  "🤖 AI Features",
        "📈  Yield Prediction":    "🤖 AI Features",
        "🌱  Crop Recommendation": "🤖 AI Features",
        "📡  IoT Sensors":         "🌾 Farm Operations",
        "🚜  Farm Management":     "🌾 Farm Operations",
        "🗺️   Live Farm Map":       "🌾 Farm Operations",
        "📋  Reports & Analytics": "🌾 Farm Operations",
        "🔔  Alert Center":        "🌾 Farm Operations",
        "🤖  AI Chat Assistant":   "💡 Intelligence",
        "⚙️   Settings":           "⚙️ System",
    }

    _last_section = "__START__"
    for opt in SELECTABLE:
        sec = _section_map.get(opt)
        if sec != _last_section:
            _last_section = sec
            if sec:
                st.markdown(f"""
<div style="font-size:9px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
            color:rgba(255,255,255,0.22);padding:1rem 1.2rem 0.25rem;
            font-family:'Inter',sans-serif">{sec}</div>
""", unsafe_allow_html=True)

        is_active = (opt == current_label)
        active_style = (
            "background:rgba(34,197,94,0.13);border-left:3px solid #22c55e;"
            "color:#4ae176;font-weight:700;"
        ) if is_active else (
            "border-left:3px solid transparent;color:rgba(255,255,255,0.55);"
        )

        # Render as clickable HTML + invisible Streamlit button on top
        col_nav = st.container()
        with col_nav:
            st.markdown(f"""
<div style="display:flex;align-items:center;{active_style}
            padding:0.58rem 1.1rem;margin:1px 8px 1px 0;
            border-radius:0 8px 8px 0;font-size:13.5px;
            font-family:'Inter',sans-serif;cursor:pointer;
            transition:all 0.2s;pointer-events:none;
            line-height:1.4">{opt}</div>
""", unsafe_allow_html=True)

            # Transparent real button placed over the div via CSS trick
            if st.button(opt, key=f"navbtn_{opt}", use_container_width=True):
                page_name = LABEL_TO_PAGE.get(opt, "Dashboard")
                st.session_state.page = page_name
                st.rerun()

    # ── User profile strip ─────────────────────────────────────────────────
    st.markdown("""
<div style="margin-top:auto;padding:1rem 1.2rem;border-top:1px solid rgba(255,255,255,0.07);
            display:flex;align-items:center;gap:0.75rem">
    <div style="width:34px;height:34px;border-radius:50%;flex-shrink:0;
                background:linear-gradient(135deg,#006e2f,#22c55e);
                display:flex;align-items:center;justify-content:center;
                font-size:14px;font-weight:800;color:white;
                box-shadow:0 0 0 2px rgba(34,197,94,0.4)">F</div>
    <div>
        <div style="font-size:12.5px;font-weight:700;color:rgba(255,255,255,0.88);
                    font-family:'Inter',sans-serif">Farmer Admin</div>
        <div style="font-size:10.5px;color:rgba(255,255,255,0.3);
                    font-family:'Inter',sans-serif">Farm Director</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE ROUTER                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════
#  1. DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    weather = get_weather()

    # Hero header
    st.markdown(f"""
<div class="page-hero animate-fadein">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem">
        <div>
            <div class="page-title">Welcome Back, Farmer 👋</div>
            <div class="page-subtitle">
                {datetime.now().strftime("%A, %B %d %Y")} &nbsp;·&nbsp;
                {weather['city']} {weather['temp']}°C {weather['desc']}
            </div>
        </div>
        <div style="display:flex;gap:.75rem;align-items:center">
            <div class="badge badge-live live-pulse">
                <span style="width:7px;height:7px;background:#22c55e;border-radius:50%;display:inline-block"></span>
                Live Monitoring
            </div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

    # ── KPI Row ────────────────────────────────────────────────────────────
    cols = st.columns(6)
    kpis = [
        ("🌾","Total Crops","12,458","+2%","green"),
        ("🚜","Active Farms","143","+5%","teal"),
        ("💧","Water Used","3,250 L","-8%","blue"),
        ("🧠","AI Health Score","94%","+1%","green"),
        ("📊","Predicted Yield","18.5 T","+12%","orange"),
        ("📡","Online Sensors","245/248",None,"purple"),
    ]
    for col, (icon, label, val, delta, color) in zip(cols, kpis):
        with col:
            st.markdown(kpi(icon, label, val, delta, color), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.75rem'>", unsafe_allow_html=True)

    # ── Main layout ────────────────────────────────────────────────────────
    left, right = st.columns([2.2, 1], gap="large")

    with left:
        # Yield & water trend chart
        st.markdown('<div class="agri-card animate-fadein-1">', unsafe_allow_html=True)
        section_title("📈 Farm Performance Overview")

        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        yield_d  = [11.2,12.1,10.8,13.5,14.2,15.8,16.4,17.1,18.5,None,None,None]
        target_d = [13]*12
        water_d  = [2800,2600,3100,3400,3250,3000,2900,3100,3250,3100,None,None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=target_d, name="Target",
            line=dict(color="#bccbb9", dash="dot", width=1.5)))
        fig.add_trace(go.Scatter(x=months, y=yield_d, name="Yield (T)",
            line=dict(color="#006e2f", width=3), fill="tonexty",
            fillcolor="rgba(0,110,47,.07)", mode="lines+markers",
            marker=dict(size=7, color="#006e2f", line=dict(width=2, color="white"))))
        fig.add_trace(go.Bar(x=months, y=water_d, name="Water (L)",
            yaxis="y2", marker_color="rgba(29,78,216,.25)",
            marker_line_color="rgba(29,78,216,.5)", marker_line_width=1))
        fig.update_layout(**plotly_defaults(), height=280,
            yaxis2=dict(overlaying="y", side="right", showgrid=False,
                        title="Water (L)", color="#6d7b6c"),
            yaxis=dict(title="Yield (T)", showgrid=True,
                       gridcolor="rgba(226,232,240,.5)", color="#6d7b6c"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Crop health donuts
        st.markdown('<div class="agri-card animate-fadein-2" style="margin-top:1rem">', unsafe_allow_html=True)
        section_title("🌿 Crop Health Analysis")

        hc = st.columns(4)
        health = [("Healthy Crops",78,"#006e2f"),("Disease Risk",12,"#ba1a1a"),
                  ("Pest Risk",6,"#f97316"),("Water Stress",4,"#3b82f6")]
        for col, (lbl, val, color) in zip(hc, health):
            with col:
                f2 = go.Figure(go.Pie(
                    values=[val,100-val], hole=0.74,
                    marker_colors=[color,"#eceef0"], textinfo="none", showlegend=False,
                    hoverinfo="skip",
                ))
                f2.update_layout(height=120, margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    annotations=[dict(text=f"<b>{val}%</b>", x=.5, y=.5,
                        font=dict(size=17, family="Inter", color="#191c1e"), showarrow=False)])
                st.plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})
                st.markdown(f"<div style='text-align:center;font-size:12px;font-weight:700;color:#191c1e;margin-top:-8px'>{lbl}</div>",
                            unsafe_allow_html=True)

        st.markdown("""
<div class="ai-box" style="margin-top:1rem">
    <div class="ai-orb">✨</div>
    <div>
        <div style="font-size:13.5px;font-weight:700;color:#006e2f;margin-bottom:.375rem">AI Insights & Recommendations</div>
        <div style="font-size:13px;color:#3d4a3d;margin-bottom:.25rem">
            🟢 <strong style="color:#191c1e">Increase irrigation in Sector B</strong> — Soil moisture dropped below 35%. Auto-schedule recommended.
        </div>
        <div style="font-size:13px;color:#3d4a3d;margin-bottom:.25rem">
            🟡 <strong style="color:#191c1e">Harvest Sector C</strong> within 4 days for optimal grain protein content (12.8%).
        </div>
        <div style="font-size:13px;color:#3d4a3d">
            🔴 <strong style="color:#191c1e">Disease Risk in Sector D-04</strong> — Brown Rust probability 86%. Apply fungicide within 24h.
        </div>
    </div>
</div>
</div>""", unsafe_allow_html=True)

    with right:
        # Current weather
        st.markdown(f"""
<div class="weather-hero animate-fadein-1">
    <div style="font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;opacity:.5;margin-bottom:.5rem">
        LIVE WEATHER · {weather['city'].upper()}
    </div>
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
            <div class="temp-big">{weather['temp']}°</div>
            <div style="font-size:14px;opacity:.7;margin-top:2px">{weather['desc']}</div>
            <div style="font-size:12px;opacity:.45;margin-top:2px">Feels like {weather['feels']}°C</div>
        </div>
        <div style="font-size:48px;margin-top:-.25rem">⛅</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:.625rem;margin-top:1.25rem">
        <div style="background:rgba(255,255,255,.07);border-radius:8px;padding:.625rem .75rem">
            <div style="font-size:10px;opacity:.45;text-transform:uppercase;letter-spacing:.05em">Humidity</div>
            <div style="font-size:18px;font-weight:700;margin-top:2px">{weather['humidity']}%</div>
        </div>
        <div style="background:rgba(255,255,255,.07);border-radius:8px;padding:.625rem .75rem">
            <div style="font-size:10px;opacity:.45;text-transform:uppercase;letter-spacing:.05em">Wind</div>
            <div style="font-size:18px;font-weight:700;margin-top:2px">{weather['wind']} km/h</div>
        </div>
        <div style="background:rgba(255,255,255,.07);border-radius:8px;padding:.625rem .75rem">
            <div style="font-size:10px;opacity:.45;text-transform:uppercase;letter-spacing:.05em">UV Index</div>
            <div style="font-size:18px;font-weight:700;margin-top:2px;color:#fbbf24">{weather['uv']} Mod</div>
        </div>
        <div style="background:rgba(255,255,255,.07);border-radius:8px;padding:.625rem .75rem">
            <div style="font-size:10px;opacity:.45;text-transform:uppercase;letter-spacing:.05em">Pressure</div>
            <div style="font-size:18px;font-weight:700;margin-top:2px">{weather['pressure']} hPa</div>
        </div>
    </div>
    <div style="display:flex;gap:6px;margin-top:1rem;overflow-x:auto;padding-bottom:4px">
        <div style="flex:1;min-width:50px;text-align:center;background:rgba(34,197,94,.18);border:1px solid rgba(34,197,94,.4);border-radius:10px;padding:.6rem .3rem">
            <div style="font-size:10px;font-weight:700;opacity:.7">Mon</div>
            <div style="font-size:20px;margin:.2rem 0">☀️</div>
            <div style="font-size:13px;font-weight:800">30°</div>
            <div style="font-size:10px;opacity:.45">20°</div>
        </div>
        <div style="flex:1;min-width:50px;text-align:center;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);border-radius:10px;padding:.6rem .3rem">
            <div style="font-size:10px;font-weight:700;opacity:.7">Tue</div>
            <div style="font-size:20px;margin:.2rem 0">☀️</div>
            <div style="font-size:13px;font-weight:800">32°</div>
            <div style="font-size:10px;opacity:.45">21°</div>
        </div>
        <div style="flex:1;min-width:50px;text-align:center;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);border-radius:10px;padding:.6rem .3rem">
            <div style="font-size:10px;font-weight:700;opacity:.7">Wed</div>
            <div style="font-size:20px;margin:.2rem 0">⛅</div>
            <div style="font-size:13px;font-weight:800">27°</div>
            <div style="font-size:10px;opacity:.45">19°</div>
        </div>
        <div style="flex:1;min-width:50px;text-align:center;background:rgba(147,197,253,.12);border:1px solid rgba(147,197,253,.3);border-radius:10px;padding:.6rem .3rem">
            <div style="font-size:10px;font-weight:700;opacity:.7">Thu</div>
            <div style="font-size:20px;margin:.2rem 0">🌧️</div>
            <div style="font-size:13px;font-weight:800">23°</div>
            <div style="font-size:10px;opacity:.45">17°</div>
            <div style="font-size:9px;color:#93c5fd;margin-top:2px">11mm</div>
        </div>
        <div style="flex:1;min-width:50px;text-align:center;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);border-radius:10px;padding:.6rem .3rem">
            <div style="font-size:10px;font-weight:700;opacity:.7">Fri</div>
            <div style="font-size:20px;margin:.2rem 0">🌦️</div>
            <div style="font-size:13px;font-weight:800">25°</div>
            <div style="font-size:10px;opacity:.45">18°</div>
        </div>
        <div style="flex:1;min-width:50px;text-align:center;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);border-radius:10px;padding:.6rem .3rem">
            <div style="font-size:10px;font-weight:700;opacity:.7">Sat</div>
            <div style="font-size:20px;margin:.2rem 0">☀️</div>
            <div style="font-size:13px;font-weight:800">29°</div>
            <div style="font-size:10px;opacity:.45">19°</div>
        </div>
        <div style="flex:1;min-width:50px;text-align:center;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);border-radius:10px;padding:.6rem .3rem">
            <div style="font-size:10px;font-weight:700;opacity:.7">Sun</div>
            <div style="font-size:20px;margin:.2rem 0">☀️</div>
            <div style="font-size:13px;font-weight:800">31°</div>
            <div style="font-size:10px;opacity:.45">20°</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

        # Alerts
        st.markdown("""
<div class="agri-card animate-fadein-2" style="margin-top:1rem">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">
        <div style="font-size:15px;font-weight:700">🔔 Alerts Center</div>
        <span class="badge badge-red">3 New</span>
    </div>
    <div class="alert alert-danger">
        <div style="display:flex;justify-content:space-between;margin-bottom:.25rem">
            <span style="font-size:12.5px;font-weight:700;color:#ba1a1a">⚠️ Disease Alert</span>
            <span style="font-size:11px;color:#6d7b6c">2m ago</span>
        </div>
        <div style="font-size:12.5px">High probability of Rust Fungi in Sector D-04.</div>
    </div>
    <div class="alert alert-warning">
        <div style="display:flex;justify-content:space-between;margin-bottom:.25rem">
            <span style="font-size:12.5px;font-weight:700;color:#c2410c">💧 Low Moisture</span>
            <span style="font-size:11px;color:#6d7b6c">1h ago</span>
        </div>
        <div style="font-size:12.5px">Sector B fallen to 32%. Irrigation triggered.</div>
    </div>
    <div class="alert alert-info">
        <div style="display:flex;justify-content:space-between;margin-bottom:.25rem">
            <span style="font-size:12.5px;font-weight:700;color:#1d4ed8">🌧️ Rain Alert</span>
            <span style="font-size:11px;color:#6d7b6c">2h ago</span>
        </div>
        <div style="font-size:12.5px">11.4mm rainfall expected Thursday. Delay fertilizer.</div>
    </div>
    <div class="alert alert-success" style="opacity:.7">
        <div style="font-size:12.5px;font-weight:700;color:#006e2f">✅ Drone Survey Complete</div>
        <div style="font-size:12.5px">Sector A-01: No anomalies detected.</div>
    </div>
</div>

<!-- Autonomous Fleet -->
<div class="agri-card animate-fadein-3" style="margin-top:1rem">
    <div style="font-size:13px;font-weight:700;color:#6d7b6c;text-transform:uppercase;letter-spacing:.06em;margin-bottom:1rem">
        🚁 Autonomous Fleet
    </div>""", unsafe_allow_html=True)

        fleet = [("✈️","Drone AG-01","Scanning Sector D",65),
                 ("🚜","Tractor TR-12","Charging (80%)",80),
                 ("🤖","Bot AGRI-5","Soil sampling",40)]
        for icon, name, status, pct in fleet:
            col_bar = "#ba1a1a" if pct < 30 else ""
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:.875rem;margin-bottom:.875rem">
    <div style="width:38px;height:38px;background:#f2f4f6;border-radius:9px;
                display:flex;align-items:center;justify-content:center;font-size:17px;flex-shrink:0">{icon}</div>
    <div style="flex:1;min-width:0">
        <div style="font-size:13px;font-weight:700">{name}</div>
        <div style="font-size:11.5px;color:#6d7b6c">{status}</div>
        {progress_bar(pct, col_bar)}
    </div>
    <div style="font-size:12px;font-weight:700;color:{'#006e2f' if pct>50 else '#f97316'};white-space:nowrap">{pct}%</div>
</div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════════════
#  2. CROP MONITORING
# ══════════════════════════════════════════════════════════════════════════
elif page == "Crop Monitoring":
    st.markdown("""
<div class="page-hero animate-fadein">
    <div class="page-title">🌿 Crop Monitoring</div>
    <div class="page-subtitle">Real-time growth tracking, health analysis, and AI-powered crop management</div>
</div>""", unsafe_allow_html=True)

    cols = st.columns(5)
    crop_kpis = [
        ("🌾","Active Crops","8 Varieties",None,"green"),
        ("📊","Avg Health Score","91%","+3%","teal"),
        ("📅","Days to Harvest","28 days",None,"orange"),
        ("🌡️","Avg Temperature","26.4°C",None,"blue"),
        ("💧","Soil Moisture","43% VWC","-2%","purple"),
    ]
    for col, (icon, label, val, delta, color) in zip(cols, crop_kpis):
        with col:
            st.markdown(kpi(icon, label, val, delta, color), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.75rem'>", unsafe_allow_html=True)
    left, right = st.columns([2, 1], gap="large")

    with left:
        section_title("🌱 Crop Growth Stages")
        crop_data = pd.DataFrame({
            "Crop":       ["Winter Wheat","Corn (Maize)","Soybean","Cotton","Rice","Lentil","Mustard","Chickpea"],
            "Stage":      ["Grain Filling","Tasseling","Pod Fill","Boll Open","Flowering","Vegetative","Pod Fill","Flowering"],
            "Health (%)": [94,88,79,92,85,76,91,83],
            "Days Left":  [14,28,35,21,42,18,24,31],
            "Area (ac)":  [120,85,60,95,140,45,70,55],
            "Status":     ["Optimal","Optimal","Watch","Optimal","Caution","Watch","Optimal","Optimal"],
        })

        def color_health(val):
            if val >= 90: return "background-color:rgba(0,110,47,.1);color:#006e2f"
            if val >= 75: return "background-color:rgba(249,115,22,.08);color:#c2410c"
            return "background-color:rgba(186,26,26,.08);color:#ba1a1a"

        st.markdown('<div class="agri-card">', unsafe_allow_html=True)
        st.markdown('<table class="ag-table"><thead><tr>', unsafe_allow_html=True)
        for c in crop_data.columns:
            st.markdown(f"<th>{c}</th>", unsafe_allow_html=True)
        st.markdown("</tr></thead><tbody>", unsafe_allow_html=True)
        for _, row in crop_data.iterrows():
            h = row["Health (%)"]
            style = color_health(h)
            s_badge = badge(row["Status"], "green" if row["Status"]=="Optimal"
                            else "orange" if row["Status"]=="Caution" else "blue")
            st.markdown(f"""<tr>
<td style="font-weight:600">{row['Crop']}</td>
<td>{row['Stage']}</td>
<td><div style="{style};border-radius:6px;padding:2px 8px;display:inline-block;font-weight:700;font-size:13px">{h}%</div></td>
<td>{row['Days Left']}d</td>
<td>{row['Area (ac)']} ac</td>
<td>{s_badge}</td>
</tr>""", unsafe_allow_html=True)
        st.markdown("</tbody></table></div>", unsafe_allow_html=True)

        # Growth trend
        st.markdown('<div class="agri-card" style="margin-top:1rem">', unsafe_allow_html=True)
        section_title("📈 Growth Trend – Last 90 Days")
        days = pd.date_range(end=datetime.today(), periods=90, freq="D")
        wheat = 10 + 40*(1/(1+np.exp(-0.08*(np.arange(90)-45)))) + np.random.randn(90)*0.8
        corn  = 8  + 35*(1/(1+np.exp(-0.07*(np.arange(90)-50)))) + np.random.randn(90)*0.9
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=wheat, name="Winter Wheat",
            line=dict(color="#006e2f", width=2.5), fill="tozeroy",
            fillcolor="rgba(0,110,47,.05)"))
        fig.add_trace(go.Scatter(x=days, y=corn, name="Corn",
            line=dict(color="#f97316", width=2.5)))
        fig.update_layout(**plotly_defaults(), height=240,
            yaxis=dict(title="Height (cm)", showgrid=True, gridcolor="rgba(226,232,240,.5)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        # AI health score
        st.markdown("""
<div class="card-gradient-green animate-fadein-1">
    <div style="font-size:10px;font-weight:700;letter-spacing:.1em;opacity:.6;text-transform:uppercase">AI HEALTH SCORE</div>
    <div style="font-size:64px;font-weight:900;line-height:1;margin:.375rem 0">91<span style="font-size:28px">%</span></div>
    <div style="font-size:13px;opacity:.8">Overall farm health index</div>
    <div style="background:rgba(255,255,255,.15);border-radius:999px;height:6px;margin-top:1rem;overflow:hidden">
        <div style="background:white;width:91%;height:100%;border-radius:999px"></div>
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:.75rem">
        <div style="text-align:center">
            <div style="font-size:18px;font-weight:800">8</div>
            <div style="font-size:10px;opacity:.5">Crops</div>
        </div>
        <div style="text-align:center">
            <div style="font-size:18px;font-weight:800">670</div>
            <div style="font-size:10px;opacity:.5">Acres</div>
        </div>
        <div style="text-align:center">
            <div style="font-size:18px;font-weight:800">245</div>
            <div style="font-size:10px;opacity:.5">Sensors</div>
        </div>
    </div>
    <div style="position:absolute;right:-10px;bottom:-10px;font-size:80px;opacity:.07;pointer-events:none">🌱</div>
</div>""", unsafe_allow_html=True)

        # Crop image upload
        st.markdown("""<div class="agri-card animate-fadein-2" style="margin-top:1rem">
            <div style="font-size:14px;font-weight:700;margin-bottom:.75rem">📷 Crop Image Analysis</div>""",
            unsafe_allow_html=True)
        img_file = st.file_uploader("Upload crop image", type=["jpg","jpeg","png"],
                                     key="crop_img", label_visibility="collapsed")
        if img_file:
            img = Image.open(img_file)
            st.image(img, use_column_width=True)
            with st.spinner("Analyzing with AI..."):
                time.sleep(0.8)
            disease, conf = predict_disease_mock(img)
            treat = DISEASE_TREATMENTS.get(disease, {})
            urg = treat.get("urgency","—")
            urg_color = "#ba1a1a" if urg=="Critical" else "#f97316" if urg=="High" else "#006e2f"
            st.markdown(f"""
<div style="margin-top:.75rem">
    <div style="display:flex;justify-content:space-between;align-items:center">
        <div style="font-size:14px;font-weight:700;color:#191c1e">{disease}</div>
        <div style="font-size:18px;font-weight:800;color:#006e2f">{conf*100:.0f}%</div>
    </div>
    {progress_bar(conf*100)}
    <div style="margin-top:.5rem;font-size:12px;color:{urg_color};font-weight:700">⚡ Urgency: {urg}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="text-align:center;padding:1.5rem;color:#6d7b6c;font-size:13px;border:2px dashed #E2E8F0;border-radius:10px">
    📷 Upload leaf/crop image for instant AI analysis
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Nutrient status
        st.markdown('<div class="agri-card animate-fadein-3" style="margin-top:1rem">', unsafe_allow_html=True)
        section_title("🧪 Nutrient Status")
        nutrients = [("Nitrogen (N)",72,"#006e2f","Adequate"),
                     ("Phosphorus (P)",45,"#3b82f6","Low"),
                     ("Potassium (K)",88,"#006e2f","Good"),
                     ("pH Balance",68,"#f97316","Slightly Acidic")]
        for name, pct, color, status in nutrients:
            st.markdown(f"""
<div style="margin-bottom:.875rem">
    <div style="display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:.25rem">
        <span style="font-weight:600">{name}</span>
        <span style="color:{color};font-weight:700">{status} ({pct}%)</span>
    </div>
    <div class="prog-track">
        <div class="prog-bar" style="width:{pct}%;background:{color}"></div>
    </div>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════════════
#  3. AI DISEASE DETECTION
# ══════════════════════════════════════════════════════════════════════════
elif page == "AI Disease Detection":
    st.markdown("""
<div class="page-hero animate-fadein">
    <div class="page-title">🔬 AI Disease Detection</div>
    <div class="page-subtitle">Upload crop images for instant pathogen identification using AgriVisionAI neural network</div>
</div>""", unsafe_allow_html=True)

    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.markdown('<div class="agri-card animate-fadein-1">', unsafe_allow_html=True)
        st.markdown("""
<div style="text-align:center;padding:1.5rem 0">
    <div style="width:72px;height:72px;background:linear-gradient(135deg,rgba(0,110,47,.1),rgba(34,197,94,.05));
                border-radius:999px;display:flex;align-items:center;justify-content:center;
                font-size:32px;margin:0 auto 1rem">🔬</div>
    <div style="font-size:18px;font-weight:700;margin-bottom:.375rem">Upload Sample Image</div>
    <div style="font-size:13px;color:#6d7b6c">Drag & drop or browse leaf/crop photos</div>
    <div style="display:flex;gap:6px;justify-content:center;margin-top:.75rem">
        <span style="background:#f2f4f6;color:#6d7b6c;border-radius:5px;padding:2px 9px;font-size:11px;font-weight:600">PNG</span>
        <span style="background:#f2f4f6;color:#6d7b6c;border-radius:5px;padding:2px 9px;font-size:11px;font-weight:600">JPG</span>
        <span style="background:#f2f4f6;color:#6d7b6c;border-radius:5px;padding:2px 9px;font-size:11px;font-weight:600">JPEG</span>
        <span style="background:#f2f4f6;color:#6d7b6c;border-radius:5px;padding:2px 9px;font-size:11px;font-weight:600">WebP</span>
    </div>
</div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["png","jpg","jpeg","webp"],
                                     key="disease_img", label_visibility="collapsed")
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_column_width=True, caption="📷 Uploaded sample")
            pbar = st.progress(0)
            for i in range(100):
                pbar.progress(i+1)
                time.sleep(0.007)
            pbar.empty()
            st.success("✅ Analysis complete!")
        else:
            st.markdown("""
<div style="border:2px dashed #bccbb9;border-radius:10px;padding:1.5rem;
            text-align:center;color:#6d7b6c;font-size:13px;margin-top:.5rem">
    👆 Click <strong>Browse files</strong> above to upload
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Detection history
        section_title("📋 Detection History")
        hist = pd.DataFrame({
            "Date":      ["Jun 18","Jun 16","Jun 14","Jun 12","Jun 10"],
            "Crop":      ["Wheat","Soybean","Maize","Rice","Cotton"],
            "Detection": ["Brown Rust 98%","Healthy","N-Deficiency 82%","Leaf Blight 76%","Powdery Mildew 91%"],
            "Severity":  ["Moderate","None","Mild","Severe","Mild"],
            "Action":    ["✅ Treated","👁 Monitor","✅ Treated","⚠️ In Progress","✅ Treated"],
        })
        sev_b = {"Moderate":"orange","None":"green","Mild":"blue","Severe":"red"}
        st.markdown('<div class="agri-card"><table class="ag-table"><thead><tr>', unsafe_allow_html=True)
        for c in hist.columns:
            st.markdown(f"<th>{c}</th>", unsafe_allow_html=True)
        st.markdown("</tr></thead><tbody>", unsafe_allow_html=True)
        for _, row in hist.iterrows():
            b = badge(row["Severity"], sev_b.get(row["Severity"],"blue"))
            st.markdown(f"<tr><td>{row['Date']}</td><td>{row['Crop']}</td>"
                        f"<td style='font-size:12px'>{row['Detection']}</td>"
                        f"<td>{b}</td><td style='font-size:13px'>{row['Action']}</td></tr>",
                        unsafe_allow_html=True)
        st.markdown("</tbody></table></div>", unsafe_allow_html=True)

    with right:
        if uploaded:
            img = Image.open(uploaded)
            disease, conf = predict_disease_mock(img)
        else:
            disease, conf = "Brown Rust", 0.982

        treat = DISEASE_TREATMENTS.get(disease, {
            "fungicide":"Consult agronomist","env":"Monitor closely","urgency":"Low"})
        urg = treat["urgency"]
        urg_color = {"Critical":"#ba1a1a","High":"#f97316","Medium":"#eab308","Low":"#006e2f"}.get(urg,"#006e2f")

        st.markdown(f"""
<div class="agri-card animate-fadein-1" style="border-left:4px solid {urg_color}">
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
            <span style="background:{urg_color}22;color:{urg_color};border-radius:999px;
                         padding:2px 10px;font-size:10px;font-weight:700;letter-spacing:.06em">
                {urg.upper()} {'🚨' if urg in ['Critical','High'] else '⚠️'}
            </span>
            <div style="font-size:26px;font-weight:800;color:#191c1e;margin-top:.5rem">🦠 {disease}</div>
            <div style="font-size:13px;color:#6d7b6c;font-style:italic;margin-top:2px">
                {next((k for k in ["Puccinia triticina","Blumeria graminis",
                 "Helminthosporium","Xanthomonas","Nutritional Disorder"]
                 if disease.split()[0] in k or k in disease), "Pathogen identified")}
            </div>
        </div>
        <div style="text-align:right">
            <div style="font-size:52px;font-weight:900;line-height:1;color:#006e2f">{conf*100:.0f}<span style="font-size:20px">%</span></div>
            <div style="font-size:10.5px;color:#6d7b6c;text-transform:uppercase;letter-spacing:.05em">Confidence</div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-top:1rem">
        <div style="background:#f7f9fb;border-radius:9px;padding:.875rem">
            <div style="font-size:10.5px;color:#6d7b6c;font-weight:600;text-transform:uppercase">Affected Area</div>
            <div style="font-size:22px;font-weight:800;margin-top:2px">4.5%</div>
        </div>
        <div style="background:#f7f9fb;border-radius:9px;padding:.875rem">
            <div style="font-size:10.5px;color:#6d7b6c;font-weight:600;text-transform:uppercase">Severity</div>
            <div style="font-size:22px;font-weight:800;color:{urg_color};margin-top:2px">{urg}</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

        # Confidence chart
        st.markdown('<div class="agri-card animate-fadein-2" style="margin-top:1rem">', unsafe_allow_html=True)
        labels = [disease, "Powdery Mildew", "Leaf Blight", "Healthy"]
        values = [conf*100, 8.2, 5.4, 3.1]
        colors = ["#006e2f","#a4f1b2","#a4f1b2","#dcfce7"]
        fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
            marker_color=colors, text=[f"{v:.1f}%" for v in values],
            textposition="outside"))
        fig.update_layout(**plotly_defaults(), height=180,
            xaxis=dict(range=[0,115], visible=False),
            yaxis=dict(autorange="reversed"), title="Detection Probabilities")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Treatment protocol
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown(f"""
<div class="agri-card animate-fadein-2">
    <div style="font-size:13.5px;font-weight:700;margin-bottom:.625rem;display:flex;gap:6px;align-items:center">
        💉 Fungicide Protocol
    </div>
    <div style="font-size:12.5px;color:#3d4a3d">
        {treat.get('fungicide','Apply broad-spectrum fungicide')}
    </div>
</div>""", unsafe_allow_html=True)
        with tc2:
            st.markdown(f"""
<div class="agri-card animate-fadein-2">
    <div style="font-size:13.5px;font-weight:700;margin-bottom:.625rem;display:flex;gap:6px;align-items:center">
        🌬️ Environment Control
    </div>
    <div style="font-size:12.5px;color:#3d4a3d">
        {treat.get('env','Monitor and maintain field hygiene')}
    </div>
</div>""", unsafe_allow_html=True)

        # AI forecast
        st.markdown(f"""
<div class="card-gradient-green animate-fadein-3" style="margin-top:0">
    <div style="display:flex;align-items:center;gap:1rem">
        <div style="background:rgba(255,255,255,.15);border-radius:999px;width:44px;height:44px;
                    display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0">✨</div>
        <div style="flex:1">
            <div style="font-weight:700;font-size:13.5px;margin-bottom:.25rem">Predictive Spread Forecast</div>
            <div style="font-size:12.5px;opacity:.85">
                Expected spread: <strong>1.2%/day</strong> if untreated.
                Isolate Sector B-4 immediately. Treatment window: <strong>24–48 hours</strong>.
            </div>
        </div>
    </div>
    <div style="position:absolute;right:-8px;bottom:-8px;font-size:60px;opacity:.07">🦠</div>
</div>""", unsafe_allow_html=True)

        # Disease heatmap
        section_title("🗺️ Disease Risk Heatmap")
        np.random.seed(42)
        Z = np.random.uniform(0, 40, (8, 10))
        Z[3:5, 7:9] = np.array([[88, 92], [78, 85]])
        fig2 = go.Figure(go.Heatmap(z=Z,
            colorscale=[[0,"#dcfce7"],[.4,"#fef9c3"],[.7,"#ffedd5"],[1,"#ba1a1a"]],
            colorbar=dict(title="Risk %", tickfont=dict(family="Inter",size=11)),
            hovertemplate="Sector [%{y},%{x}]<br>Risk: %{z:.0f}%<extra></extra>"))
        fig2.update_layout(**plotly_defaults(), height=200,
            xaxis=dict(title="Column"), yaxis=dict(title="Row"))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    footer()


# ══════════════════════════════════════════════════════════════════════════
#  4. SMART IRRIGATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "Smart Irrigation":
    st.markdown("""
<div class="page-hero animate-fadein">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem">
        <div>
            <div class="page-title">💧 Smart Irrigation & IoT</div>
            <div class="page-subtitle">Real-time pump control, valve management and soil monitoring for Farm Block A-12</div>
        </div>
        <div class="badge badge-live live-pulse" style="font-size:12px;padding:6px 16px">
            ✨ AI Optimization Active
        </div>
    </div>
</div>""", unsafe_allow_html=True)

    cols = st.columns(4)
    irr_kpis=[("💧","Tank Level","1,440 L (72%)",None,"blue"),
              ("🌱","Soil Moisture","45% VWC","-2%","green"),
              ("⚙️","Pump Flow","42 L/min",None,"orange"),
              ("📡","Active Sensors","18/18","","teal")]
    for col, (icon,label,val,delta,color) in zip(cols, irr_kpis):
        with col: st.markdown(kpi(icon,label,val,delta,color), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.75rem'>", unsafe_allow_html=True)
    left, right = st.columns([2.2, 1], gap="large")

    with left:
        top = st.columns(2)
        with top[0]:
            st.markdown('<div class="agri-card animate-fadein-1">', unsafe_allow_html=True)
            st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem">
    <div>
        <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#6d7b6c">Water Tank Level</div>
        <div style="font-size:22px;font-weight:800;margin-top:4px">72% Capacity</div>
    </div>
    <div style="background:#dbeafe;border-radius:9px;padding:8px;font-size:20px">💧</div>
</div>""", unsafe_allow_html=True)
            fig_tank = go.Figure(go.Pie(values=[72,28], hole=0.72,
                marker_colors=["#3b82f6","#eceef0"], textinfo="none", showlegend=False))
            fig_tank.update_layout(height=160, margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(text="<b>1,440L</b><br><span>Left</span>",x=.5,y=.5,
                    font=dict(size=14,family="Inter"),showarrow=False)])
            st.plotly_chart(fig_tank, use_container_width=True, config={"displayModeBar":False})
            st.markdown("""
<div style="display:flex;justify-content:space-between;font-size:12px;color:#6d7b6c;
            border-top:1px solid #E2E8F0;padding-top:.625rem">
    <span>Last refill: 4h ago</span>
    <span style="color:#006e2f;font-weight:700">✓ Safe Range</span>
</div></div>""", unsafe_allow_html=True)

        with top[1]:
            st.markdown('<div class="agri-card animate-fadein-1">', unsafe_allow_html=True)
            st.markdown("""
<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#6d7b6c;margin-bottom:.5rem">Soil Moisture – 12h History</div>
""", unsafe_allow_html=True)
            hrs = list(range(-11, 1))
            mois = [36,37,39,41,43,44,43,45,46,44,45,45]
            fig_m = go.Figure(go.Scatter(x=hrs, y=mois, fill="tozeroy",
                fillcolor="rgba(0,110,47,.07)", line=dict(color="#006e2f",width=2.5),
                mode="lines+markers", marker=dict(size=5,color="#006e2f")))
            fig_m.add_hline(y=35, line_dash="dot", line_color="#ba1a1a",
                            annotation_text="Critical", annotation_position="right")
            fig_m.add_hline(y=60, line_dash="dot", line_color="#3b82f6",
                            annotation_text="Saturated", annotation_position="right")
            fig_m.update_layout(**plotly_defaults(), height=155,
                xaxis=dict(title="Hours ago",showgrid=False),
                yaxis=dict(showgrid=True,gridcolor="rgba(226,232,240,.5)",
                           title="% VWC",range=[20,75]))
            st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar":False})
            st.markdown("""
<div style="font-size:12px;color:#3d4a3d;text-align:center">
    💡 AI suggests irrigation in <strong style="color:#006e2f">3 hours</strong>
</div></div>""", unsafe_allow_html=True)

        # Pump & valve control
        st.markdown('<div class="agri-card animate-fadein-2" style="margin-top:1rem">', unsafe_allow_html=True)
        pl, pr = st.columns([2,1])
        with pl:
            st.markdown("""
<div style="display:flex;align-items:center;gap:.875rem">
    <div style="width:48px;height:48px;background:rgba(0,110,47,.1);border-radius:999px;
                display:flex;align-items:center;justify-content:center;font-size:22px">⚙️</div>
    <div>
        <div style="font-size:18px;font-weight:700">Main Pump System</div>
        <div style="display:flex;align-items:center;gap:6px;color:#006e2f;font-weight:700;font-size:13px">
            <span style="width:8px;height:8px;background:#22c55e;border-radius:50%;
                         display:inline-block;animation:pulse-green 2s infinite"></span>
            Running Normally · 42 L/min
        </div>
    </div>
</div>""", unsafe_allow_html=True)
        with pr:
            pump = st.toggle("Pump Active", value=True, key="pump_toggle")
            st.markdown(f"""
<div style="font-size:13.5px;font-weight:700;color:{'#006e2f' if pump else '#ba1a1a'}">
    {'● RUNNING' if pump else '○ STOPPED'}
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1rem'><strong style='font-size:13.5px'>🔧 Sector Valve Controls</strong></div>", unsafe_allow_html=True)
        v1,v2,v3,v4 = st.columns(4)
        sectors_ctrl = [(v1,"North-01",True),(v2,"North-02",True),(v3,"South-01",False),(v4,"South-02",False)]
        for col, name, dflt in sectors_ctrl:
            with col:
                on = st.toggle(f"Sector {name}", value=dflt, key=f"valve_{name}")
                color = "#006e2f" if on else "#6d7b6c"
                st.markdown(f"""
<div style="text-align:center;font-size:11px;color:{color};font-weight:700">
    {'🟢 OPEN' if on else '⚫ CLOSED'}
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Schedule table
        section_title("📅 Irrigation Schedule – Next 7 Days")
        sched = pd.DataFrame({
            "Sector":   ["North-01","North-02","South-01","South-02","North-01","All Sectors","South-02"],
            "Date":     ["Jun 20","Jun 21","Jun 22","Jun 23","Jun 24","Jun 25","Jun 26"],
            "Time":     ["06:00 AM","06:30 AM","05:45 AM","07:00 AM","06:00 AM","05:30 AM","07:00 AM"],
            "Duration": ["45 min","30 min","60 min","45 min","30 min","90 min","35 min"],
            "Volume":   ["1,890 L","1,260 L","2,520 L","1,890 L","1,260 L","5,040 L","1,470 L"],
            "Status":   ["✅ Done","✅ Done","⏳ Scheduled","⏳ Scheduled","⏳ Scheduled","⏳ Scheduled","⏳ Scheduled"],
        })
        st.markdown('<div class="agri-card"><table class="ag-table"><thead><tr>', unsafe_allow_html=True)
        for c in sched.columns: st.markdown(f"<th>{c}</th>", unsafe_allow_html=True)
        st.markdown("</tr></thead><tbody>", unsafe_allow_html=True)
        for _, row in sched.iterrows():
            done = row["Status"].startswith("✅")
            st.markdown(f"<tr><td style='font-weight:600'>{row['Sector']}</td>"
                        f"<td>{row['Date']}</td><td>{row['Time']}</td>"
                        f"<td>{row['Duration']}</td><td>{row['Volume']}</td>"
                        f"<td style='color:{'#006e2f' if done else '#6d7b6c'}'>{row['Status']}</td></tr>",
                        unsafe_allow_html=True)
        st.markdown("</tbody></table></div>", unsafe_allow_html=True)

    with right:
        weather = get_weather()
        st.markdown(f"""
<div class="weather-hero animate-fadein-1" style="border-radius:var(--radius-lg)">
    <div style="font-size:10px;font-weight:700;letter-spacing:.1em;opacity:.45;text-transform:uppercase">LOCAL ATMOSPHERE</div>
    <div class="temp-big" style="font-size:56px;margin:.375rem 0">{weather['temp']}°C</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;margin-top:1rem">
        <div style="background:rgba(255,255,255,.07);border-radius:7px;padding:.5rem .625rem;text-align:center">
            <div style="font-size:10px;opacity:.45">Humidity</div>
            <div style="font-weight:700;font-size:15px">{weather['humidity']}%</div>
        </div>
        <div style="background:rgba(255,255,255,.07);border-radius:7px;padding:.5rem .625rem;text-align:center">
            <div style="font-size:10px;opacity:.45">Wind</div>
            <div style="font-weight:700;font-size:15px">{weather['wind']}km/h</div>
        </div>
        <div style="background:rgba(255,255,255,.07);border-radius:7px;padding:.5rem .625rem;text-align:center">
            <div style="font-size:10px;opacity:.45">UV</div>
            <div style="font-weight:700;font-size:15px;color:#fbbf24">{weather['uv']}</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="card-gradient-green animate-fadein-2" style="margin-top:1rem">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:.75rem">
        <span style="font-size:14px">⚡</span>
        <span style="font-size:10.5px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;opacity:.7">
            AI IRRIGATION INSIGHT
        </span>
    </div>
    <div style="font-size:13.5px;line-height:1.65;margin-bottom:1.25rem;opacity:.92">
        Increase water supply by <strong>12%</strong> in Sector North-01.
        Tomorrow's heat wave will raise evapotranspiration by <strong>22%</strong>.
        Rain Thursday — skip scheduled Friday cycle.
    </div>
    <div style="position:absolute;right:-8px;bottom:-8px;font-size:60px;opacity:.07">💧</div>
</div>""", unsafe_allow_html=True)
        if st.button("⚡ Execute AI Recommendation", use_container_width=True, key="exec_irr"):
            st.success("✅ Schedule updated by AI!")

        section_title("📡 IoT Sensor Grid")
        sensors = [
            ("S-102","Soil Moisture","45.2%","2m","🌱",True),
            ("T-45","Temperature","24.8°C","2m","🌡️",True),
            ("H-88","Humidity","58%","1m","💧",True),
            ("P-12","pH Level","6.4","5m","🧪",True),
            ("W-05","Water Pressure","2.4 bar","Active","⚡",True),
            ("N-22","Nitrogen","18 mg/kg","10m","🌿",True),
        ]
        for i in range(0, len(sensors), 2):
            s_cols = st.columns(2)
            for j, (sid,label,val,ts,icon,online) in enumerate(sensors[i:i+2]):
                with s_cols[j]:
                    status_dot = "sensor-online" if online else "sensor-offline"
                    st.markdown(f"""
<div class="sensor-chip" style="margin-bottom:.5rem">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.25rem">
        <span style="font-size:11px;font-weight:700;color:#6d7b6c">{sid}</span>
        <span class="{status_dot}"></span>
    </div>
    <div style="font-size:11px;color:#6d7b6c">{icon} {label}</div>
    <div style="font-size:20px;font-weight:800;margin-top:2px">{val}</div>
    <div style="font-size:10px;color:#6d7b6c;opacity:.6;margin-top:2px">{ts} ago</div>
</div>""", unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════════════
#  5. WEATHER ANALYTICS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Weather Analytics":
    weather = get_weather()
    st.markdown("""
<div class="page-hero animate-fadein">
    <div class="page-title">🌤️ Weather Analytics</div>
    <div class="page-subtitle">Real-time atmospheric data and AI-powered weather-to-yield correlation analysis</div>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="weather-hero animate-fadein-1">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1.5rem;position:relative;z-index:1">
        <div style="display:flex;align-items:center;gap:1.5rem">
            <div style="background:rgba(34,197,94,.12);border-radius:999px;padding:1.25rem;font-size:48px">⛅</div>
            <div>
                <div style="display:flex;align-items:baseline;gap:.75rem">
                    <span class="temp-big">{weather['temp']}°C</span>
                    <span style="font-size:20px;opacity:.6;font-weight:500">{weather['desc']}</span>
                </div>
                <div style="font-size:13px;opacity:.45;margin-top:4px">
                    Feels like {weather['feels']}°C · {weather['city']} ·
                    {datetime.now().strftime('%b %d, %Y %H:%M')}
                </div>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem">
            <div style="text-align:center">
                <div style="font-size:10px;opacity:.4;text-transform:uppercase;letter-spacing:.06em;font-weight:700">💧 Humidity</div>
                <div style="font-size:24px;font-weight:800;margin-top:4px">{weather['humidity']}%</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:10px;opacity:.4;text-transform:uppercase;letter-spacing:.06em;font-weight:700">💨 Wind</div>
                <div style="font-size:24px;font-weight:800;margin-top:4px">{weather['wind']} km/h</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:10px;opacity:.4;text-transform:uppercase;letter-spacing:.06em;font-weight:700">☀️ UV</div>
                <div style="font-size:24px;font-weight:800;margin-top:4px;color:#fbbf24">{weather['uv']}</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:10px;opacity:.4;text-transform:uppercase;letter-spacing:.06em;font-weight:700">⏱️ Pressure</div>
                <div style="font-size:24px;font-weight:800;margin-top:4px">{weather['pressure']}</div>
            </div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

    f_col, r_col = st.columns([1.2, 1], gap="large")
    with f_col:
        section_title("📅 7-Day Forecast")
        st.markdown('<div class="agri-card animate-fadein-2">', unsafe_allow_html=True)
        st.markdown("""
<div style="display:flex;gap:6px;width:100%;overflow-x:auto;">
        <div style="flex:1;min-width:72px;text-align:center;background:rgba(0,110,47,.12);border:1px solid rgba(0,110,47,.3);
                    border-radius:10px;padding:.7rem .4rem;">
            <div style="font-size:11px;font-weight:700;color:#6d7b6c">Mon</div>
            <div style="font-size:24px;margin:.3rem 0">☀️</div>
            <div style="font-size:16px;font-weight:800">30°</div>
            <div style="font-size:11px;color:#6d7b6c;margin-top:1px">20°</div>
        </div>
        <div style="flex:1;min-width:72px;text-align:center;background:rgba(255,255,255,.7);border:1px solid #E2E8F0;
                    border-radius:10px;padding:.7rem .4rem;">
            <div style="font-size:11px;font-weight:700;color:#6d7b6c">Tue</div>
            <div style="font-size:24px;margin:.3rem 0">☀️</div>
            <div style="font-size:16px;font-weight:800">32°</div>
            <div style="font-size:11px;color:#6d7b6c;margin-top:1px">21°</div>
        </div>
        <div style="flex:1;min-width:72px;text-align:center;background:rgba(255,255,255,.7);border:1px solid #E2E8F0;
                    border-radius:10px;padding:.7rem .4rem;">
            <div style="font-size:11px;font-weight:700;color:#6d7b6c">Wed</div>
            <div style="font-size:24px;margin:.3rem 0">⛅</div>
            <div style="font-size:16px;font-weight:800">27°</div>
            <div style="font-size:11px;color:#6d7b6c;margin-top:1px">19°</div>
            <div style="font-size:10px;color:#3b82f6;font-weight:600;margin-top:3px">2.1mm</div>
        </div>
        <div style="flex:1;min-width:72px;text-align:center;background:rgba(29,78,216,.05);border:1px solid #E2E8F0;
                    border-radius:10px;padding:.7rem .4rem;">
            <div style="font-size:11px;font-weight:700;color:#6d7b6c">Thu</div>
            <div style="font-size:24px;margin:.3rem 0">🌧️</div>
            <div style="font-size:16px;font-weight:800">23°</div>
            <div style="font-size:11px;color:#6d7b6c;margin-top:1px">17°</div>
            <div style="font-size:10px;color:#3b82f6;font-weight:600;margin-top:3px">11.4mm</div>
        </div>
        <div style="flex:1;min-width:72px;text-align:center;background:rgba(255,255,255,.7);border:1px solid #E2E8F0;
                    border-radius:10px;padding:.7rem .4rem;">
            <div style="font-size:11px;font-weight:700;color:#6d7b6c">Fri</div>
            <div style="font-size:24px;margin:.3rem 0">🌦️</div>
            <div style="font-size:16px;font-weight:800">25°</div>
            <div style="font-size:11px;color:#6d7b6c;margin-top:1px">18°</div>
            <div style="font-size:10px;color:#3b82f6;font-weight:600;margin-top:3px">3.2mm</div>
        </div>
        <div style="flex:1;min-width:72px;text-align:center;background:rgba(255,255,255,.7);border:1px solid #E2E8F0;
                    border-radius:10px;padding:.7rem .4rem;">
            <div style="font-size:11px;font-weight:700;color:#6d7b6c">Sat</div>
            <div style="font-size:24px;margin:.3rem 0">☀️</div>
            <div style="font-size:16px;font-weight:800">29°</div>
            <div style="font-size:11px;color:#6d7b6c;margin-top:1px">19°</div>
        </div>
        <div style="flex:1;min-width:72px;text-align:center;background:rgba(255,255,255,.7);border:1px solid #E2E8F0;
                    border-radius:10px;padding:.7rem .4rem;">
            <div style="font-size:11px;font-weight:700;color:#6d7b6c">Sun</div>
            <div style="font-size:24px;margin:.3rem 0">☀️</div>
            <div style="font-size:16px;font-weight:800">31°</div>
            <div style="font-size:11px;color:#6d7b6c;margin-top:1px">20°</div>
        </div>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r_col:
        section_title("🌧️ Rainfall Forecast (mm)")
        rain_days = [d for d,_,_,_,_ in FORECAST_MOCK]
        rain_vals  = [r for _,_,_,_,r in FORECAST_MOCK]
        fig_rain = go.Figure(go.Bar(x=rain_days, y=rain_vals,
            marker_color=["#3b82f6" if v > 5 else "#93c5fd" if v > 0 else "#e2e8f0" for v in rain_vals],
            text=[f"{v}mm" if v > 0 else "" for v in rain_vals], textposition="outside"))
        fig_rain.update_layout(**plotly_defaults(), height=200,
            yaxis=dict(title="mm", showgrid=True, gridcolor="rgba(226,232,240,.5)"))
        st.plotly_chart(fig_rain, use_container_width=True, config={"displayModeBar": False})

    # Temperature trend
    section_title("🌡️ Temperature Trends – Last 30 Days")
    st.markdown('<div class="agri-card animate-fadein-3">', unsafe_allow_html=True)
    dates = pd.date_range(end=datetime.today(), periods=30, freq="D")
    day_temp  = 22 + 8*np.sin(np.linspace(0, 2*np.pi, 30)) + np.random.randn(30)*1.2
    night_temp= day_temp - 7 + np.random.randn(30)*0.8
    avg_line  = np.full(30, 25.5)
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=dates, y=night_temp, name="Night Temp",
        line=dict(color="#93c5fd",dash="dot"), fill=None))
    fig_t.add_trace(go.Scatter(x=dates, y=avg_line, name="30-day Avg",
        line=dict(color="#f97316",dash="dash",width=1.5)))
    fig_t.add_trace(go.Scatter(x=dates, y=day_temp, name="Day Temp",
        line=dict(color="#006e2f",width=2.5), fill="tonexty",
        fillcolor="rgba(0,110,47,.05)"))
    fig_t.update_layout(**plotly_defaults(), height=240,
        yaxis=dict(title="°C", showgrid=True, gridcolor="rgba(226,232,240,.5)"))
    st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Weather advisories
    section_title("⚡ Weather-Based Farm Advisories")
    a1, a2, a3 = st.columns(3)
    advisories = [
        (a1,"🌧️ Rain Alert – Thursday",
         "11.4mm expected. Delay fertilizer 2 days. Pre-harvest soil compaction risk.","orange"),
        (a2,"☀️ Ideal Harvest Window",
         "Mon–Wed: humidity <60%, zero rain. Perfect for Sector C mechanical harvest.","green"),
        (a3,"💨 Spray Advisory",
         "Wind speeds 14 km/h Friday. Postpone pesticide spraying to Saturday morning.","blue"),
    ]
    for col, title, body, color in advisories:
        with col:
            bg = {"orange":"rgba(249,115,22,.04)","green":"rgba(0,110,47,.04)","blue":"rgba(29,78,216,.04)"}[color]
            border = {"orange":"#f97316","green":"#006e2f","blue":"#1d4ed8"}[color]
            st.markdown(f"""
<div style="background:{bg};border:1px solid {border}22;border-left:3px solid {border};
            border-radius:0 var(--radius-md) var(--radius-md) 0;padding:1rem">
    <div style="font-size:13.5px;font-weight:700;margin-bottom:.375rem;color:#191c1e">{title}</div>
    <div style="font-size:12.5px;color:#3d4a3d">{body}</div>
</div>""", unsafe_allow_html=True)
    footer()

# ══════════════════════════════════════════════════════════════════════════
#  6. YIELD PREDICTION
# ══════════════════════════════════════════════════════════════════════════
elif page == "Yield Prediction":
    st.markdown("""
<div class="page-hero animate-fadein">
    <div class="page-title">📈 Yield Prediction</div>
    <div class="page-subtitle">AI harvest forecasting with 91% accuracy — powered by 1.2M historical data points</div>
</div>""", unsafe_allow_html=True)

    left, right = st.columns([1, 2.2], gap="large")

    with left:
        st.markdown('<div class="agri-card animate-fadein-1">', unsafe_allow_html=True)
        section_title("🎛️ Prediction Inputs")
        crop   = st.selectbox("Crop Type", ["Corn (Zea mays)","Soybeans","Winter Wheat","Alfalfa","Rice","Cotton"])
        soil   = st.selectbox("Soil Type", ["Loamy (High Organic)","Sandy Clay","Silt Loam"])
        area   = st.number_input("Farm Area (Acres)", 1, 10000, 120)
        season = st.selectbox("Season", ["Kharif 2026","Rabi 2026-27","Zaid 2026"])
        base_yield = {"Corn (Zea mays)":18.5,"Soybeans":12.3,"Winter Wheat":15.8,
                      "Alfalfa":22.1,"Rice":16.4,"Cotton":9.8}.get(crop, 15.0)
        soil_m = {"Loamy (High Organic)":1.1,"Sandy Clay":0.85,"Silt Loam":0.97}.get(soil, 1.0)
        pred_yield = round(base_yield * soil_m * (area / 100), 2)
        st.markdown("""
<div class="ai-box" style="margin-top:.75rem;font-size:12.5px">
    <div class="ai-orb" style="width:32px;height:32px;font-size:14px">🔄</div>
    <div><strong style="color:#006e2f">Weather Auto-synced</strong><br>
    <span style="color:#3d4a3d">Station #442 + Global GFS model active</span></div>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("✨ Recalculate AI Model", use_container_width=True, key="recalc_yield"):
            with st.spinner("Updating model…"):
                time.sleep(1.2)
            st.success("Model recalculated!")

        st.markdown(f"""
<div class="card-gradient-green animate-fadein-2" style="margin-top:1rem">
    <div style="font-size:10px;font-weight:700;letter-spacing:.1em;opacity:.6;text-transform:uppercase">AI CONFIDENCE</div>
    <div style="font-size:56px;font-weight:900;line-height:1;margin:.25rem 0">91%</div>
    <div style="font-size:12.5px;opacity:.8">Based on 1.2M data points for this region</div>
    <div style="background:rgba(255,255,255,.2);border-radius:999px;height:6px;margin-top:.875rem;overflow:hidden">
        <div style="background:white;width:91%;height:100%;border-radius:999px"></div>
    </div>
    <div style="position:absolute;right:-8px;bottom:-8px;font-size:60px;opacity:.07">🧠</div>
</div>""", unsafe_allow_html=True)

    with right:
        k_cols = st.columns(4)
        revenue = round(pred_yield * 2270)
        for col, (icon,lbl,val,delta,color) in zip(k_cols, [
            ("🌾","Est. Yield",f"{pred_yield:.1f} T","+12%","green"),
            ("📅","Harvest Date","Oct 12, 2026",None,"blue"),
            ("💰","Revenue",f"${revenue:,}",None,"orange"),
            ("📊","Profit Margin","32%","+3%","teal"),
        ]):
            with col: st.markdown(kpi(icon,lbl,val,delta,color), unsafe_allow_html=True)

        # Projection curve
        st.markdown('<div class="agri-card animate-fadein-2" style="margin-top:1rem">', unsafe_allow_html=True)
        months = ["Jun","Jul","Aug","Sep","Oct","Nov"]
        predicted_c = [2.1,5.4,9.8,14.2,pred_yield,pred_yield*0.92]
        historical  = [1.9,4.8,8.9,12.5,15.1,14.2]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months,y=historical,name="Historical Avg",
            line=dict(color="#bccbb9",dash="dash",width=2),fill=None))
        fig.add_trace(go.Scatter(x=months,y=predicted_c,name="AI Prediction",
            line=dict(color="#006e2f",width=3),
            fill="tonexty",fillcolor="rgba(0,110,47,.07)",
            mode="lines+markers",marker=dict(size=8,color="#006e2f",
                line=dict(width=2,color="white"))))
        fig.add_annotation(x="Sep",y=14.2,text="<b>Peak: 14.2T</b>",
            showarrow=True,arrowhead=2,arrowcolor="#006e2f",
            bgcolor="white",bordercolor="#006e2f",borderwidth=1,
            font=dict(family="Inter",size=11,color="#006e2f"))
        fig.update_layout(**plotly_defaults(),height=270,
            title="Yield Projection Curve",
            yaxis=dict(title="Tons",showgrid=True,gridcolor="rgba(226,232,240,.5)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Feature importance & year comparison
        fi_col, yc_col = st.columns(2)
        with fi_col:
            st.markdown('<div class="agri-card animate-fadein-3" style="margin-top:1rem">', unsafe_allow_html=True)
            features = ["Rainfall","Soil Quality","Temperature","Fertilizer","Irrigation","Pest Ctrl"]
            imp      = [0.28,0.24,0.18,0.14,0.10,0.06]
            fig2 = go.Figure(go.Bar(x=imp, y=features, orientation="h",
                marker_color=["#006e2f","#22c55e","#4ade80","#86efac","#bbf7d0","#dcfce7"],
                text=[f"{v*100:.0f}%" for v in imp], textposition="outside"))
            fig2.update_layout(**plotly_defaults(), height=220,
                title="Feature Importance",
                xaxis=dict(range=[0,.35],visible=False),
                yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
            st.markdown("</div>", unsafe_allow_html=True)

        with yc_col:
            st.markdown('<div class="agri-card animate-fadein-3" style="margin-top:1rem">', unsafe_allow_html=True)
            years = [2021,2022,2023,2024,2025,2026]
            actual   = [14.2,15.1,13.8,16.4,17.2,None]
            forecast = [None,None,None,None,17.2,pred_yield]
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=years,y=actual,name="Actual",marker_color="#006e2f",opacity=0.85))
            fig3.add_trace(go.Bar(x=years,y=forecast,name="Forecast",marker_color="#a4f1b2",opacity=0.9))
            fig3.update_layout(**plotly_defaults(),height=220,title="Year-on-Year",
                barmode="group",xaxis=dict(showgrid=False),
                yaxis=dict(title="T",showgrid=True,gridcolor="rgba(226,232,240,.5)"))
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
            st.markdown("</div>", unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════════════
#  7. CROP RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "Crop Recommendation":
    st.markdown("""
<div class="page-hero animate-fadein">
    <div class="page-title">🌱 AI Crop Recommendation</div>
    <div class="page-subtitle">Enter soil & climate parameters — our ML model suggests the best crops with nutrient analysis</div>
</div>""", unsafe_allow_html=True)

    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.markdown('<div class="agri-card animate-fadein-1">', unsafe_allow_html=True)
        section_title("🧪 Soil & Environment Parameters")
        c1,c2,c3 = st.columns(3)
        with c1: N = st.number_input("Nitrogen (N)", 0, 200, 90, help="mg/kg")
        with c2: P = st.number_input("Phosphorus (P)", 0, 200, 42, help="mg/kg")
        with c3: K = st.number_input("Potassium (K)", 0, 200, 43, help="mg/kg")
        c4,c5 = st.columns(2)
        with c4: temp  = st.slider("Temperature (°C)", 0.0, 50.0, 26.0, 0.5)
        with c5: humid = st.slider("Humidity (%)", 0.0, 100.0, 65.0, 0.5)
        c6,c7 = st.columns(2)
        with c6: ph       = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1)
        with c7: rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 202.9)

        run = st.button("🔍 Get AI Recommendation", use_container_width=True, key="run_crop")
        st.markdown("</div>", unsafe_allow_html=True)

        # Nutrient analysis
        nut_stat = nutrient_status(N, P, K, ph)
        st.markdown('<div class="agri-card animate-fadein-2" style="margin-top:1rem">', unsafe_allow_html=True)
        section_title("🧪 Nutrient Analysis")
        nut_vals = {"Nitrogen (N)":N/140*100, "Phosphorus (P)":P/75*100,
                    "Potassium (K)":K/100*100, "pH Balance":(ph/14)*100}
        nut_colors = {"Optimal":"#006e2f","Deficient":"#ba1a1a","Excess":"#f97316"}
        for nut, (status, color) in nut_stat.items():
            pct = min(nut_vals.get(nut, 50), 100)
            st.markdown(f"""
<div style="margin-bottom:.875rem">
    <div style="display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:.25rem">
        <span style="font-weight:600">{nut}</span>
        <span style="color:{color};font-weight:700">{status}</span>
    </div>
    <div class="prog-track">
        <div class="prog-bar" style="width:{pct:.0f}%;background:{color}"></div>
    </div>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        crop_name, conf = predict_crop_model(N, P, K, temp, humid, ph, rainfall)
        crop_icons = {"Rice":"🌾","Maize":"🌽","Wheat":"🌾","Cotton":"🌱",
                      "Coffee":"☕","Mango":"🥭","Banana":"🍌","Rice":"🍚"}
        c_icon = crop_icons.get(crop_name, "🌿")

        st.markdown(f"""
<div class="agri-card animate-fadein-1" style="border-left:4px solid #006e2f">
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
            <span class="badge badge-green" style="font-size:10px;letter-spacing:.06em">AI RECOMMENDATION</span>
            <div style="font-size:30px;font-weight:900;color:#006e2f;margin-top:.5rem">{c_icon} {crop_name}</div>
            <div style="font-size:13px;color:#6d7b6c;margin-top:2px">Best match for your soil & climate</div>
        </div>
        <div style="text-align:right">
            <div style="font-size:52px;font-weight:900;line-height:1;color:#006e2f">{conf*100:.0f}<span style="font-size:20px">%</span></div>
            <div style="font-size:10.5px;color:#6d7b6c;text-transform:uppercase;letter-spacing:.05em">Confidence</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

        # Top 5 alternatives
        alts = [(crop_name,conf),(CROP_NAMES[(CROP_NAMES.index(crop_name)+3)%len(CROP_NAMES)],0.78),
                (CROP_NAMES[(CROP_NAMES.index(crop_name)+7)%len(CROP_NAMES)],0.71),
                (CROP_NAMES[(CROP_NAMES.index(crop_name)+11)%len(CROP_NAMES)],0.64),
                (CROP_NAMES[(CROP_NAMES.index(crop_name)+15)%len(CROP_NAMES)],0.55)]
        names  = [a[0] for a in alts]
        scores = [a[1]*100 for a in alts]
        st.markdown('<div class="agri-card animate-fadein-2" style="margin-top:1rem">', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=scores, y=names, orientation="h",
            marker_color=["#006e2f","#22c55e","#4ade80","#86efac","#dcfce7"],
            text=[f"{s:.0f}%" for s in scores], textposition="outside"))
        fig.update_layout(**plotly_defaults(), height=200, title="Top 5 Crop Suitability",
            xaxis=dict(range=[0,115],visible=False), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Growing tips & fertilizer
        tips_map = {
            "Rice":   ["Requires waterlogged soil","Ideal temp: 20–35°C","Harvest: 90–150 days"],
            "Maize":  ["Well-drained fertile soil","Ideal temp: 18–27°C","Harvest: 60–100 days"],
            "Wheat":  ["Loamy soil preferred","Ideal temp: 12–24°C","Harvest: 110–140 days"],
            "Cotton": ["Sandy loam preferred","Ideal temp: 21–30°C","Harvest: 150–180 days"],
        }
        tips = tips_map.get(crop_name, ["Ensure proper drainage","Monitor soil moisture","Apply balanced fertilizer"])
        fertilizer = {"Nitrogen (N)":"120 kg/ha","Phosphorus (P)":"60 kg/ha","Potassium (K)":"40 kg/ha","Zinc":"5 kg/ha"}

        t_col, f_col = st.columns(2)
        with t_col:
            st.markdown(f"""
<div class="agri-card animate-fadein-3">
    <div style="font-size:13.5px;font-weight:700;margin-bottom:.625rem">🌱 Growing Tips</div>
    {"".join(f'<div style="font-size:12.5px;color:#3d4a3d;display:flex;gap:6px;margin-bottom:.375rem"><span style=color:#006e2f;font-weight:700>✓</span>{t}</div>' for t in tips)}
</div>""", unsafe_allow_html=True)
        with f_col:
            st.markdown('<div class="agri-card animate-fadein-3">', unsafe_allow_html=True)
            st.markdown("<div style='font-size:13.5px;font-weight:700;margin-bottom:.625rem'>🧪 Fertilizer Plan</div>",
                        unsafe_allow_html=True)
            for k, v in fertilizer.items():
                st.markdown(f"""
<div style="display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:.375rem">
    <span style="color:#3d4a3d">{k}</span>
    <span style="font-weight:700;color:#006e2f">{v}</span>
</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════════════
#  8. IOT SENSORS
# ══════════════════════════════════════════════════════════════════════════
elif page == "IoT Sensors":
    st.markdown("""
<div class="page-hero animate-fadein">
    <div class="page-title">📡 IoT Sensor Management</div>
    <div class="page-subtitle">Real-time telemetry from 248 distributed sensors across all farm sectors</div>
</div>""", unsafe_allow_html=True)

    cols = st.columns(4)
    for col, (icon,lbl,val,delta,color) in zip(cols,[
        ("📡","Total Sensors","248",None,"green"),
        ("✅","Online","245",None,"teal"),
        ("⚠️","Low Battery","3",None,"orange"),
        ("🔴","Offline","0",None,"blue"),
    ]):
        with col: st.markdown(kpi(icon,lbl,val,delta,color), unsafe_allow_html=True)

    section_title("🔬 Live Sensor Grid")
    all_sensors = [
        ("S-102","Soil Moisture","45.2%","🌱","#006e2f",True,"2m","N-01"),
        ("S-103","Soil Moisture","38.1%","🌱","#f97316",True,"2m","N-02"),
        ("S-104","Soil Moisture","61.4%","🌱","#3b82f6",True,"3m","S-01"),
        ("T-45", "Temperature","24.8°C","🌡️","#191c1e",True,"2m","N-01"),
        ("T-46", "Temperature","26.2°C","🌡️","#191c1e",True,"1m","S-01"),
        ("H-88", "Humidity","58%","💧","#191c1e",True,"1m","N-01"),
        ("H-89", "Humidity","62%","💧","#191c1e",True,"2m","S-02"),
        ("P-12", "pH Level","6.4","🧪","#7c3aed",True,"5m","N-01"),
        ("P-13", "pH Level","6.8","🧪","#7c3aed",True,"4m","S-01"),
        ("W-05", "Water Pressure","2.4 bar","⚡","#0d9488",True,"Active","Main"),
        ("W-06", "Water Pressure","2.1 bar","⚡","#0d9488",True,"Active","S-01"),
        ("N-22", "Nitrogen","18 mg/kg","🌿","#006e2f",True,"10m","N-01"),
        ("N-23", "Nitrogen","12 mg/kg","🌿","#f97316",True,"8m","S-02"),
        ("L-01", "Light Intensity","42k lux","☀️","#eab308",True,"1m","N-01"),
        ("L-02", "Light Intensity","38k lux","☀️","#eab308",True,"1m","S-01"),
        ("CO2-1","CO₂ Level","412 ppm","💨","#6d7b6c",True,"5m","N-01"),
        ("EC-01","Conductivity","1.8 mS/cm","⚗️","#7c3aed",True,"12m","N-02"),
        ("GW-01","Groundwater","3.2 m","🏔️","#0284c7",True,"30m","Farm"),
    ]

    sensor_filter = st.selectbox("Filter by Sector",
        ["All Sectors","N-01","N-02","S-01","S-02","Main","Farm"], key="sensor_filter")
    filtered = [s for s in all_sensors if sensor_filter == "All Sectors" or s[7] == sensor_filter]

    cols_per_row = 6
    for i in range(0, len(filtered), cols_per_row):
        g_cols = st.columns(cols_per_row)
        for j, (sid,label,val,icon,color,online,ts,sector) in enumerate(filtered[i:i+cols_per_row]):
            with g_cols[j]:
                dot = "sensor-online" if online else "sensor-offline"
                st.markdown(f"""
<div class="sensor-chip animate-fadein" style="margin-bottom:.625rem">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.375rem">
        <span style="font-size:11px;font-weight:800;color:#6d7b6c">{sid}</span>
        <span class="{dot}"></span>
    </div>
    <div style="font-size:11px;color:#6d7b6c;margin-bottom:.25rem">{icon} {label}</div>
    <div style="font-size:19px;font-weight:800;color:{color};line-height:1.2">{val}</div>
    <div style="display:flex;justify-content:space-between;margin-top:.375rem">
        <span style="font-size:10px;color:#6d7b6c;opacity:.6">{ts} ago</span>
        <span style="font-size:10px;color:#6d7b6c;opacity:.6">{sector}</span>
    </div>
</div>""", unsafe_allow_html=True)

    # Sensor trend chart
    section_title("📈 Sensor Trends – Last 24 Hours")
    st.markdown('<div class="agri-card animate-fadein-2">', unsafe_allow_html=True)
    hours24 = pd.date_range(end=datetime.now(), periods=24, freq="h")
    moist  = 40+10*np.sin(np.linspace(0,2*np.pi,24))+np.random.randn(24)*1.5
    temp24 = 22+8*np.sin(np.linspace(0.5,2.5*np.pi,24))+np.random.randn(24)*0.8
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours24,y=moist,name="Soil Moisture (%)",
        line=dict(color="#006e2f",width=2.5),fill="tozeroy",fillcolor="rgba(0,110,47,.05)"))
    fig.add_trace(go.Scatter(x=hours24,y=temp24,name="Temperature (°C)",
        line=dict(color="#f97316",width=2.5),yaxis="y2"))
    fig.update_layout(**plotly_defaults(),height=250,
        yaxis=dict(title="Moisture %",showgrid=True,gridcolor="rgba(226,232,240,.5)"),
        yaxis2=dict(overlaying="y",side="right",title="Temp °C",color="#f97316",showgrid=False))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)
    footer()


# ══════════════════════════════════════════════════════════════════════════
#  9. FARM MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════
elif page == "Farm Management":
    st.markdown("""
<div class="page-hero animate-fadein">
    <div class="page-title">🚜 Farm Management</div>
    <div class="page-subtitle">Multi-farm registration, crop allocation, and field management dashboard</div>
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏠 My Farms", "➕ Add Farm", "📊 Farm Analytics"])

    with tab1:
        farms = [
            ("🌾 Green Valley Farm","Tuscany, Italy","500 ac","Wheat, Corn","Optimal","#006e2f",94),
            ("🌱 North Field Block","Punjab, India","280 ac","Rice, Mustard","Caution","#f97316",72),
            ("🌿 South Plantation","Odisha, India","350 ac","Cotton, Jute","Optimal","#006e2f",89),
            ("🍃 East Orchards","Karnataka, India","180 ac","Mango, Coconut","Optimal","#006e2f",96),
        ]
        for name,loc,area,crops,status,color,score in farms:
            st.markdown(f"""
<div class="agri-card animate-fadein" style="margin-bottom:1rem">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem">
        <div style="display:flex;align-items:center;gap:1rem">
            <div style="width:48px;height:48px;background:rgba(0,110,47,.1);border-radius:12px;
                        display:flex;align-items:center;justify-content:center;font-size:22px">🏡</div>
            <div>
                <div style="font-size:16px;font-weight:700">{name}</div>
                <div style="font-size:13px;color:#6d7b6c;margin-top:2px">📍 {loc} · 📐 {area} · 🌿 {crops}</div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:1.5rem">
            <div style="text-align:center">
                <div style="font-size:10px;color:#6d7b6c;text-transform:uppercase;font-weight:700">AI Score</div>
                <div style="font-size:22px;font-weight:800;color:{color}">{score}%</div>
            </div>
            <span style="background:{color}22;color:{color};border-radius:999px;padding:4px 14px;
                         font-size:12px;font-weight:700">{status}</span>
        </div>
    </div>
    <div style="margin-top:1rem">
        <div style="display:flex;justify-content:space-between;font-size:11px;color:#6d7b6c;margin-bottom:4px">
            <span>Health Score</span><span>{score}%</span>
        </div>
        {progress_bar(score)}
    </div>
</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="agri-card animate-fadein">', unsafe_allow_html=True)
        section_title("➕ Register New Farm")
        fc1,fc2 = st.columns(2)
        with fc1:
            st.text_input("Farm Name", placeholder="e.g. East Field Block")
            st.text_input("Location / Village", placeholder="e.g. Cuttack, Odisha")
            st.number_input("Total Area (Acres)", 1, 10000, 100)
        with fc2:
            st.selectbox("Primary Crop", CROP_NAMES)
            st.selectbox("Soil Type", ["Loamy","Sandy Clay","Silt Loam","Black Cotton Soil","Red Laterite"])
            st.text_input("GPS Coordinates", placeholder="20.2961° N, 85.8245° E")
        if st.button("✅ Register Farm", key="reg_farm"):
            st.success("🎉 Farm registered successfully!")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="agri-card animate-fadein">', unsafe_allow_html=True)
        farm_names  = ["Green Valley","North Field","South Plantation","East Orchards"]
        farm_yields = [18.5,12.3,15.8,22.1]
        farm_water  = [3250,2100,2800,1900]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=farm_names,y=farm_yields,name="Yield (T)",marker_color="#006e2f"))
        fig.add_trace(go.Bar(x=farm_names,y=[w/1000 for w in farm_water],name="Water (kL)",
                              marker_color="#3b82f6",yaxis="y2"))
        fig.update_layout(**plotly_defaults(),height=280,barmode="group",
            yaxis=dict(title="Yield (T)"),
            yaxis2=dict(overlaying="y",side="right",title="Water (kL)",showgrid=False))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════════════
#  10. LIVE FARM MAP
# ══════════════════════════════════════════════════════════════════════════
elif page == "Live Farm Map":
    st.markdown("""
<div class="page-hero animate-fadein">
    <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
            <div class="page-title">🗺️ Live Farm Map</div>
            <div class="page-subtitle">Interactive GPS monitoring of all sectors, sensors and assets</div>
        </div>
        <div class="badge badge-live live-pulse" style="font-size:12px;padding:6px 16px">
            ● Live GPS Tracking
        </div>
    </div>
</div>""", unsafe_allow_html=True)

    map_col, info_col = st.columns([2.2, 1], gap="large")

    with map_col:
        st.markdown("""
<div class="farm-map-container animate-fadein-1">
    <div class="map-grid"></div>
    <!-- Sector overlays -->
    <div style="position:absolute;top:15%;left:10%;width:35%;height:28%;
                background:rgba(0,110,47,.15);border:2px solid rgba(34,197,94,.4);
                border-radius:8px;display:flex;align-items:center;justify-content:center">
        <span style="color:rgba(74,225,118,.8);font-weight:700;font-size:13px">Sector A · 120 ac</span>
    </div>
    <div style="position:absolute;top:15%;left:48%;width:38%;height:28%;
                background:rgba(249,115,22,.1);border:2px solid rgba(249,115,22,.4);
                border-radius:8px;display:flex;align-items:center;justify-content:center">
        <span style="color:rgba(251,146,60,.9);font-weight:700;font-size:13px">Sector B ⚠️ 85 ac</span>
    </div>
    <div style="position:absolute;top:50%;left:10%;width:35%;height:28%;
                background:rgba(0,110,47,.12);border:2px solid rgba(34,197,94,.3);
                border-radius:8px;display:flex;align-items:center;justify-content:center">
        <span style="color:rgba(74,225,118,.8);font-weight:700;font-size:13px">Sector C · 95 ac</span>
    </div>
    <div style="position:absolute;top:50%;left:48%;width:38%;height:28%;
                background:rgba(186,26,26,.08);border:2px solid rgba(186,26,26,.35);
                border-radius:8px;display:flex;align-items:center;justify-content:center">
        <span style="color:rgba(248,113,113,.9);font-weight:700;font-size:13px">Sector D 🚨 60 ac</span>
    </div>
    <!-- Map pins -->
    <div class="map-pin" style="top:27%;left:27%"></div>
    <div class="map-pin" style="top:27%;left:65%;background:#f97316;box-shadow:0 0 0 4px rgba(249,115,22,.3)"></div>
    <div class="map-pin" style="top:63%;left:27%"></div>
    <div class="map-pin" style="top:63%;left:65%;background:#ba1a1a;box-shadow:0 0 0 4px rgba(186,26,26,.3)"></div>
    <!-- Drone position -->
    <div style="position:absolute;top:40%;left:55%;font-size:18px;animation:fadeIn 1s infinite alternate">✈️</div>
    <!-- Labels -->
    <div class="map-label" style="top:25%;left:10%">📡 S-102: 45% VWC</div>
    <div class="map-label" style="top:55%;left:48%">⚠️ Disease Risk: 86%</div>
    <!-- Overlay card -->
    <div class="map-overlay">
        <div style="font-size:10px;opacity:.6;text-transform:uppercase;letter-spacing:.06em">SECTOR A-12</div>
        <div style="font-size:13.5px;font-weight:700;margin-top:4px;color:#4ae176">✓ Optimal</div>
        <div style="font-size:12px;opacity:.75;margin-top:2px">Moisture: 42% · Temp: 24°C</div>
        <div style="font-size:12px;opacity:.75">Crop: Organic Wheat</div>
    </div>
    <!-- Controls -->
    <div class="map-controls">
        <div class="map-ctrl-btn">+</div>
        <div class="map-ctrl-btn">−</div>
        <div class="map-ctrl-btn">⊙</div>
        <div class="map-ctrl-btn">⤢</div>
    </div>
    <!-- Legend -->
    <div style="position:absolute;top:1rem;left:1rem;background:rgba(0,0,0,.6);border-radius:8px;
                padding:.625rem .875rem;backdrop-filter:blur(8px);border:1px solid rgba(34,197,94,.2)">
        <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,.5);text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:.375rem">LEGEND</div>
        <div style="font-size:11px;color:white;display:flex;align-items:center;gap:6px;margin-bottom:3px">
            <span style="width:8px;height:8px;background:#22c55e;border-radius:50%;display:inline-block"></span> Optimal
        </div>
        <div style="font-size:11px;color:white;display:flex;align-items:center;gap:6px;margin-bottom:3px">
            <span style="width:8px;height:8px;background:#f97316;border-radius:50%;display:inline-block"></span> Caution
        </div>
        <div style="font-size:11px;color:white;display:flex;align-items:center;gap:6px">
            <span style="width:8px;height:8px;background:#ba1a1a;border-radius:50%;display:inline-block"></span> Alert
        </div>
    </div>
</div>""", unsafe_allow_html=True)

    with info_col:
        st.markdown("""
<div class="agri-card animate-fadein-2">
    <div style="font-size:14px;font-weight:700;margin-bottom:1rem">📍 Sector Status</div>""", unsafe_allow_html=True)
        sectors_info = [
            ("A","Optimal","120 ac","Wheat","🟢","#006e2f"),
            ("B","Low Moisture","85 ac","Corn","🟡","#f97316"),
            ("C","Harvest Ready","95 ac","Wheat","🟢","#006e2f"),
            ("D","Disease Alert","60 ac","Maize","🔴","#ba1a1a"),
        ]
        for sec, status, area, crop, dot, color in sectors_info:
            st.markdown(f"""
<div style="background:#f7f9fb;border-radius:9px;padding:.875rem;margin-bottom:.625rem;
            border-left:3px solid {color}">
    <div style="display:flex;justify-content:space-between;align-items:center">
        <div style="font-size:14px;font-weight:700">Sector {sec}</div>
        <span style="color:{color};font-size:12px;font-weight:700">{dot} {status}</span>
    </div>
    <div style="font-size:12px;color:#6d7b6c;margin-top:3px">{area} · {crop}</div>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
<div class="agri-card animate-fadein-3" style="margin-top:1rem">
    <div style="font-size:14px;font-weight:700;margin-bottom:1rem">🚁 Asset Tracker</div>
    <div style="font-size:13px;color:#3d4a3d;margin-bottom:.5rem">✈️ <strong>Drone AG-01</strong> — Sector D scan</div>
    <div style="font-size:13px;color:#3d4a3d;margin-bottom:.5rem">🚜 <strong>Tractor TR-12</strong> — Charging bay</div>
    <div style="font-size:13px;color:#3d4a3d;margin-bottom:.5rem">🤖 <strong>Bot AGRI-5</strong> — Sector B sampling</div>
    <div style="font-size:13px;color:#3d4a3d">🚁 <strong>Drone AG-02</strong> — Sector A scan</div>
</div>""", unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════════════
#  11. REPORTS & ANALYTICS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Reports & Analytics":
    st.markdown("""
<div class="page-hero animate-fadein">
    <div class="page-title">📋 Reports & Analytics</div>
    <div class="page-subtitle">Comprehensive farm performance analytics, export-ready reports and historical trends</div>
</div>""", unsafe_allow_html=True)

    # 1. Define summary data early so export buttons have scope access to it
    summary = pd.DataFrame({
        "Crop":          ["Winter Wheat","Corn","Soybean","Cotton","Rice"],
        "Area (ac)":     [120,85,60,95,140],
        "Yield (T)":     [18.5,12.3,8.7,14.2,22.1],
        "Revenue ($K)":  [120,85,45,92,140],
        "Health Score":  [94,88,79,92,85],
        "Status":        ["Optimal","Optimal","Watch","Optimal","Caution"],
    })

    # 2. Configured native file download streaming handlers
    btn_col1, btn_col2, btn_col3, _ = st.columns([1,1,1,3])
    with btn_col1:
        # FIXED: Updated helper function with explicit keywords compatible across fpdf / fpdf2 versions
        def generate_pdf_report(df):
            from fpdf import FPDF
            
            pdf = FPDF(orientation='P', unit='mm', format='A4')
            pdf.add_page()
            
            # --- Document Header ---
            pdf.set_font("Helvetica", style="B", size=18)
            pdf.cell(w=0, h=12, text="Agrivion AI - Farm Performance Report", border=0, ln=1, align="C")
            
            pdf.set_font("Helvetica", style="I", size=10)
            current_time = datetime.now().strftime('%b %d, %Y %H:%M')
            pdf.cell(w=0, h=8, text=f"Generated: {current_time}", border=0, ln=1, align="C")
            pdf.ln(10)
            
            # --- Section Title ---
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(w=0, h=10, text="1. Crop Summary Metrics Table", border=0, ln=1)
            pdf.ln(4)
            
            # --- Table Generation ---
            widths = [45, 25, 25, 30, 30, 25] # Total fits within A4 boundaries (~180mm)
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.set_fill_color(240, 244, 241) # Theme background fill accent
            
            # Print Columns Headers
            for i, heading in enumerate(df.columns):
                pdf.cell(w=widths[i], h=9, text=str(heading), border=1, align="C", fill=True)
            pdf.ln()
            
            # Print Body Rows
            pdf.set_font("Helvetica", style="", size=10)
            for _, row in df.iterrows():
                pdf.cell(w=widths[0], h=8, text=str(row["Crop"]), border=1)
                pdf.cell(w=widths[1], h=8, text=str(row["Area (ac)"]), border=1, align="R")
                pdf.cell(w=widths[2], h=8, text=str(row["Yield (T)"]), border=1, align="R")
                pdf.cell(w=widths[3], h=8, text=f"${row['Revenue ($K)']}K", border=1, align="R")
                pdf.cell(w=widths[4], h=8, text=f"{row['Health Score']}%", border=1, align="C")
                pdf.cell(w=widths[5], h=8, text=str(row["Status"]), border=1, align="C")
                pdf.ln()
                
            return pdf.output()

        # Safely wrap internal stream compiler output into python byte stream
        pdf_bytes = bytes(generate_pdf_report(summary))

        st.download_button(
            label="⬇ Export PDF", 
            data=pdf_bytes,
            file_name="farm_performance_analytics.pdf",
            mime="application/pdf",
            key="exp_pdf"
        )
        
    with btn_col2:
        # Dynamically converts data framework directly to browser CSV string stream
        csv_data = summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇ Export CSV",
            data=csv_data,
            file_name="farm_performance_metrics.csv",
            mime="text/csv",
            key="exp_csv"
        )
        
    with btn_col3:
        if st.button("📧 Email Report", key="email_rep"): 
            st.toast("Report emailed!", icon="✉️")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Yield & Revenue","💧 Water Usage","🦠 Disease Analytics","🌿 Crop Summary"])

    with tab1:
        r1, r2 = st.columns([1.5,1], gap="large")
        with r1:
            st.markdown('<div class="agri-card animate-fadein">', unsafe_allow_html=True)
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            yields2025 = [10.2,11.5,12.1,13.8,14.2,15.8,16.4,17.1,18.5,None,None,None]
            yields2024 = [9.1, 10.2,11.0,12.5,13.1,14.2,15.0,15.8,16.2,16.8,15.4,14.2]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months,y=yields2024,name="2024",
                line=dict(color="#bccbb9",dash="dash",width=2)))
            fig.add_trace(go.Scatter(x=months,y=yields2025,name="2025 (current)",
                line=dict(color="#006e2f",width=3),fill="tonexty",
                fillcolor="rgba(0,110,47,.06)",mode="lines+markers",
                marker=dict(size=7,color="#006e2f")))
            fig.update_layout(**plotly_defaults(),height=260,title="Annual Yield Comparison (Tons)",
                yaxis=dict(title="Tons",showgrid=True,gridcolor="rgba(226,232,240,.5)"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            st.markdown("</div>", unsafe_allow_html=True)

        with r2:
            st.markdown("""
<div class="card-gradient-green animate-fadein">
    <div style="font-size:10px;font-weight:700;letter-spacing:.1em;opacity:.6;text-transform:uppercase">Total Revenue YTD</div>
    <div style="font-size:48px;font-weight:900;line-height:1;margin:.375rem 0">$284K</div>
    <div style="font-size:13px;opacity:.8">+18% vs. same period last year</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-top:1.25rem">
        <div style="background:rgba(255,255,255,.1);border-radius:9px;padding:.875rem;text-align:center">
            <div style="font-size:10px;opacity:.5">Net Profit</div>
            <div style="font-size:20px;font-weight:800;margin-top:2px">$91K</div>
        </div>
        <div style="background:rgba(255,255,255,.1);border-radius:9px;padding:.875rem;text-align:center">
            <div style="font-size:10px;opacity:.5">Margin</div>
            <div style="font-size:20px;font-weight:800;margin-top:2px">32%</div>
        </div>
    </div>
    <div style="position:absolute;right:-8px;bottom:-8px;font-size:60px;opacity:.07">💰</div>
</div>""", unsafe_allow_html=True)

            rev_by_crop = {"Wheat":120,"Corn":85,"Cotton":45,"Rice":34}
            fig2 = go.Figure(go.Pie(labels=list(rev_by_crop.keys()),
                values=list(rev_by_crop.values()), hole=0.6,
                marker_colors=["#006e2f","#22c55e","#4ade80","#86efac"]))
            fig2.update_layout(**plotly_defaults(), height=200, showlegend=True,
                title="Revenue by Crop ($K)",
                annotations=[dict(text="<b>$284K</b>",x=.5,y=.5,
                    font=dict(size=14,family="Inter"),showarrow=False)])
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    with tab2:
        days14 = pd.date_range(end=datetime.today(), periods=14, freq="D")
        usage14 = [3100,2800,3200,3500,2900,3100,3300,3000,3250,3100,2950,3400,3100,3250]
        target14 = [3000]*14
        fig = go.Figure()
        fig.add_trace(go.Bar(x=days14, y=usage14, name="Actual",
            marker_color=["#ba1a1a" if u>3300 else "#006e2f" for u in usage14]))
        fig.add_trace(go.Scatter(x=days14, y=target14, name="Target",
            line=dict(color="#f97316",dash="dash",width=2)))
        fig.update_layout(**plotly_defaults(),height=280,title="Daily Water Consumption (Litres)",
            yaxis=dict(title="Litres",showgrid=True,gridcolor="rgba(226,232,240,.5)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with tab3:
        d_dates = pd.date_range(end=datetime.today(), periods=30, freq="D")
        detections  = np.random.poisson(2, 30)
        treated     = np.clip(detections - np.random.poisson(0.5,30), 0, None)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=d_dates, y=detections, name="Detected",marker_color="#ba1a1a",opacity=0.8))
        fig.add_trace(go.Bar(x=d_dates, y=treated, name="Treated",marker_color="#006e2f",opacity=0.8))
        fig.update_layout(**plotly_defaults(),height=280,barmode="overlay",
            title="Disease Detections vs. Treatments (30 Days)",
            yaxis=dict(title="Count",showgrid=True,gridcolor="rgba(226,232,240,.5)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with tab4:
        table_html = '<div class="agri-card animate-fadein"><table class="ag-table"><thead><tr>'
        for header in summary.columns:
            table_html += f"<th>{header}</th>"
        table_html += "</tr></thead><tbody>"
        
        sev_b = {"Optimal":"green","Watch":"blue","Caution":"orange"}
        for _, row in summary.iterrows():
            b = badge(row["Status"], sev_b.get(row["Status"],"green"))
            table_html += (
                f"<tr><td style='font-weight:600'>{row['Crop']}</td>"
                f"<td>{row['Area (ac)']}</td><td>{row['Yield (T)']}</td>"
                f"<td>${row['Revenue ($K)']}K</td>"
                f"<td><strong style='color:#006e2f'>{row['Health Score']}%</strong></td>"
                f"<td>{b}</td></tr>"
            )
        table_html += "</tbody></table></div>"
        
        st.markdown(table_html, unsafe_allow_html=True)

    footer()
# ══════════════════════════════════════════════════════════════════════════
#  13. AI CHAT ASSISTANT
# ══════════════════════════════════════════════════════════════════════════
elif page == "AI Chat Assistant":
    st.markdown("""
<div class="page-hero animate-fadein">
    <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
            <div class="page-title">🤖 AI Chat Assistant</div>
            <div class="page-subtitle">Ask anything about crops, diseases, irrigation, weather — in your language</div>
        </div>
        <div class="badge badge-live live-pulse" style="font-size:12px;padding:6px 16px">
            🟢 AgriBot Online
        </div>
    </div>
</div>""", unsafe_allow_html=True)

    main_chat, side_chat = st.columns([2, 1], gap="large")

    with side_chat:
        st.markdown("""
<div class="agri-card animate-fadein-1">
    <div style="font-size:13.5px;font-weight:700;margin-bottom:.875rem">🌐 Language & Settings</div>""",
            unsafe_allow_html=True)
        lang = st.selectbox("Response Language",
            ["English","Hindi","Odia","Bengali","Telugu","Tamil","Kannada","Marathi","Punjabi","Gujarati"],
            key="chat_lang_sel")
        st.session_state.chat_lang = lang
        if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True):
            st.session_state.chat_history = [{"role":"assistant","content":
                "👋 Hello! I'm **AgriBot**, your AI farming assistant. Ask me anything!"}]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
<div class="agri-card animate-fadein-2" style="margin-top:1rem">
    <div style="font-size:13.5px;font-weight:700;margin-bottom:.875rem">⚡ Quick Questions</div>""",
            unsafe_allow_html=True)
        quick_qs = [
            "What's the best crop for my soil?",
            "How to treat Brown Rust?",
            "When should I irrigate?",
            "Pest prevention tips for Kharif?",
            "What fertilizer does wheat need?",
            "How to improve soil pH?",
        ]
        for q in quick_qs:
            if st.button(f"💬 {q[:35]}…" if len(q)>35 else f"💬 {q}",
                         key=f"quick_{hashlib.md5(q.encode()).hexdigest()[:6]}",
                         use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":q})
                with st.spinner("AgriBot thinking..."):
                    reply = groq_chat(st.session_state.chat_history, lang)
                st.session_state.chat_history.append({"role":"assistant","content":reply})
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
<div class="card-gradient-green animate-fadein-3" style="margin-top:1rem">
    <div style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;opacity:.6;margin-bottom:.5rem">AgriBot Capabilities</div>
    <div style="font-size:12.5px;opacity:.85;line-height:1.8">
        ✅ Crop disease diagnosis<br>
        ✅ Irrigation scheduling<br>
        ✅ Yield optimization<br>
        ✅ Weather advisories<br>
        ✅ Soil & nutrient advice<br>
        ✅ 10+ Indian languages<br>
        ✅ 24/7 farmer support
    </div>
    <div style="position:absolute;right:-8px;bottom:-8px;font-size:60px;opacity:.07">🤖</div>
</div>""", unsafe_allow_html=True)

    with main_chat:
        # Chat display
        chat_html = '<div class="chat-container" id="chat-box">'
        for msg in st.session_state.chat_history:
            ts = datetime.now().strftime("%H:%M")
            if msg["role"] == "user":
                content = msg["content"].replace("\n","<br>")
                chat_html += f'<div class="msg-user">{content}<div class="msg-time">{ts}</div></div>'
            else:
                content = msg["content"].replace("\n","<br>").replace("**","<strong>").replace("**","</strong>")
                # simple bold parse
                import re
                content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', msg["content"])
                content = content.replace("\n","<br>")
                chat_html += f'<div class="msg-bot">{content}<div class="msg-time">{ts}</div></div>'
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

        # Input area
        inp_col, btn_col = st.columns([5, 1])
        with inp_col:
            user_input = st.text_input("",
                placeholder="Ask about crops, diseases, irrigation, weather...",
                key="chat_input", label_visibility="collapsed")
        with btn_col:
            send = st.button("Send 📤", use_container_width=True, key="send_btn")

        if (send or user_input) and user_input.strip():
            st.session_state.chat_history.append({"role":"user","content":user_input.strip()})
            with st.spinner(""):
                reply = groq_chat(st.session_state.chat_history, st.session_state.chat_lang)
            st.session_state.chat_history.append({"role":"assistant","content":reply})
            st.rerun()

    footer()


# ══════════════════════════════════════════════════════════════════════════
#  14. SETTINGS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Settings":
    st.markdown("""
<div class="page-hero animate-fadein">
    <div class="page-title">⚙️ Settings</div>
    <div class="page-subtitle">Configure API keys, AI models, alerts and system preferences</div>
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🔑 API Keys","🤖 AI Models","🔔 Alerts","👤 Profile"])

    with tab1:
        st.markdown('<div class="agri-card animate-fadein">', unsafe_allow_html=True)
        section_title("🔑 API Configuration")
        groq_key_status = "✅ Configured" if GROQ_API_KEY and GROQ_API_KEY!="your_groq_api_key_here" else "⚠️ Not Set"
        wx_key_status   = "✅ Configured" if WEATHER_API_KEY and WEATHER_API_KEY!="your_openweather_api_key_here" else "⚠️ Not Set"

        st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem">
    <div style="background:#f7f9fb;border-radius:9px;padding:1rem">
        <div style="font-size:12px;font-weight:700;color:#6d7b6c">Groq AI API</div>
        <div style="font-size:14px;font-weight:700;margin-top:4px">AgriBot Chat</div>
        <div style="font-size:12px;margin-top:4px;color:{'#006e2f' if '✅' in groq_key_status else '#f97316'}">{groq_key_status}</div>
    </div>
    <div style="background:#f7f9fb;border-radius:9px;padding:1rem">
        <div style="font-size:12px;font-weight:700;color:#6d7b6c">OpenWeather API</div>
        <div style="font-size:14px;font-weight:700;margin-top:4px">Live Weather</div>
        <div style="font-size:12px;margin-top:4px;color:{'#006e2f' if '✅' in wx_key_status else '#f97316'}">{wx_key_status}</div>
    </div>
</div>""", unsafe_allow_html=True)
        new_groq = st.text_input("Groq API Key", type="password",
                                  placeholder="gsk_…", value=GROQ_API_KEY if GROQ_API_KEY != "your_groq_api_key_here" else "")
        new_wx   = st.text_input("OpenWeather API Key", type="password",
                                  placeholder="abc123…", value=WEATHER_API_KEY if WEATHER_API_KEY != "your_openweather_api_key_here" else "")
        if st.button("💾 Save API Keys", key="save_api"):
            env_content = f"""GROQ_API_KEY={new_groq}\nOPENWEATHER_API_KEY={new_wx}\nAPP_NAME=Agrivion AI\nAPP_VERSION=2.0.0\n"""
            with open(".env","w") as f: f.write(env_content)
            st.success("✅ API keys saved to .env!")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="agri-card animate-fadein">', unsafe_allow_html=True)
        section_title("🤖 AI Model Configuration")
        d_model_path = os.path.exists(MODEL_PATH)
        c_model_path = os.path.exists(CROP_MODEL_PATH)
        st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
    <div style="background:#f7f9fb;border-radius:9px;padding:1rem">
        <div style="font-size:12px;font-weight:700;color:#6d7b6c">Disease Model</div>
        <div style="font-size:13px;font-weight:600;margin-top:4px">AgriVisionAI_Final.h5</div>
        <div style="font-size:11px;margin-top:4px;color:{'#006e2f' if d_model_path else '#ba1a1a'}">
            {'✅ Loaded' if d_model_path else '❌ Not Found'}
        </div>
    </div>
    <div style="background:#f7f9fb;border-radius:9px;padding:1rem">
        <div style="font-size:12px;font-weight:700;color:#6d7b6c">Crop Model</div>
        <div style="font-size:13px;font-weight:600;margin-top:4px">crop_model.pkl</div>
        <div style="font-size:11px;margin-top:4px;color:{'#006e2f' if c_model_path else '#ba1a1a'}">
            {'✅ Loaded' if c_model_path else '❌ Not Found'}
        </div>
    </div>
</div>""", unsafe_allow_html=True)
        uploaded_model = st.file_uploader("Upload Model File (.h5 or .pkl)", type=["h5","pkl"], key="model_upload")
        if uploaded_model:
            dest = f"models/{uploaded_model.name}"
            with open(dest,"wb") as f: f.write(uploaded_model.read())
            st.success(f"✅ Model saved to {dest}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="agri-card animate-fadein">', unsafe_allow_html=True)
        section_title("🔔 Alert Preferences")
        c1,c2 = st.columns(2)
        with c1:
            st.toggle("Disease Alerts", value=True)
            st.toggle("Low Moisture Alerts", value=True)
            st.toggle("Pest Alerts", value=True)
        with c2:
            st.toggle("Heavy Rain Alerts", value=True)
            st.toggle("Sensor Failure Alerts", value=True)
            st.toggle("Harvest Advisory", value=True)
        st.slider("Alert Sensitivity", 1, 10, 7)
        st.text_input("Alert Email", placeholder="farmer@example.com")
        if st.button("💾 Save Alert Settings", key="save_alerts"):
            st.success("✅ Alert settings saved!")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="agri-card animate-fadein">', unsafe_allow_html=True)
        section_title("👤 Farmer Profile")
        p1,p2 = st.columns(2)
        with p1:
            st.text_input("Full Name", value="Farmer Admin")
            st.text_input("Phone", placeholder="+91 9876543210")
            st.selectbox("Primary Language", ["English","Hindi","Odia","Bengali"])
        with p2:
            st.text_input("Email", placeholder="farmer@agrivion.ai")
            st.text_input("Village / Location", placeholder="Cuttack, Odisha")
            st.selectbox("Farm Size Category", ["Small (<5 ac)","Medium (5-50 ac)","Large (>50 ac)"])
        if st.button("💾 Save Profile", key="save_profile"):
            st.success("✅ Profile updated!")
        st.markdown("</div>", unsafe_allow_html=True)

    footer()