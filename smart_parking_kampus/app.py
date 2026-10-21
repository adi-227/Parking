import os
import sys
import re
import cv2
import threading
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import importlib
import parking_logic
import ocr_engine
import panduan_view
importlib.reload(parking_logic)
importlib.reload(ocr_engine)
importlib.reload(panduan_view)

from parking_logic import (
    normalize_plate,
    format_plate_display,
    disambiguate_plate,
    format_duration,
    decide,
    rupiah,
    TARIFFS,
    CATEGORIES,
    is_valid_plate_pattern,
    is_plausible_plate_box,
    extract_candidate_plate_boxes,
    SmoothBoxTracker,
    calculate_parking_fee,
)
import db_manager
from dataset_validator import validate_dataset
from panduan_view import render_panduan_ai

# ==============================================================================
# INISIALISASI SISTEM & DATABASE
# ==============================================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(APP_DIR, "models", "license_plate_detector.pt")
DATASET_DIR = os.path.join(APP_DIR, "dataset")

st.set_page_config(
    page_title="Smart Parking Kampus — Sistem Deteksi Plat Nomor",
    page_icon="🅿️",
    layout="wide",
)

db_manager.init_db()

# ==============================================================================
# TEMA & GAYA VISUAL (PREMIUM DARK THEME — CAMPUS DASHBOARD)
# ==============================================================================
theme_vars = """
    --bg-canvas: #0E1117;
    --bg-card: #161B26;
    --bg-card-hover: #1E2433;
    --bg-sidebar: #11151F;
    --border-color: rgba(255, 255, 255, 0.14);
    --border-card: rgba(255, 255, 255, 0.18);
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --tab-list-bg: #161B26;
    --tab-bg-inactive: #11151F;
    --tab-border-inactive: rgba(255, 255, 255, 0.14);
    --tab-text-inactive: #E2E8F0;
    --input-bg: #161B26;
    --input-text: #F8FAFC;
    --input-border: rgba(255, 255, 255, 0.2);
    --alert-info-bg: rgba(30, 58, 138, 0.25);
    --alert-info-text: #93C5FD;
    --alert-info-border: rgba(59, 130, 246, 0.4);
    --alert-success-bg: rgba(16, 185, 129, 0.15);
    --alert-success-text: #6EE7B7;
    --alert-success-border: rgba(16, 185, 129, 0.4);
    --alert-warning-bg: rgba(245, 158, 11, 0.15);
    --alert-warning-text: #FDE68A;
    --alert-warning-border: rgba(245, 158, 11, 0.4);
"""

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}

    :root {{
        {theme_vars}
        --background-color: var(--bg-canvas);
        --secondary-background-color: var(--bg-card);
        --text-color: var(--text-primary);
    }}

    .main .block-container {{
        max-width: 1320px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
    }}

    /* ==========================================================================
       CANVAS & GLOBAL TEXT CONTRAST (SOLUSI UTAMA LIGHT MODE)
       ========================================================================== */
    .stApp {{
        background-color: var(--bg-canvas) !important;
        color: var(--text-primary) !important;
    }}

    header[data-testid="stHeader"] {{
        background-color: var(--bg-canvas) !important;
    }}
    header[data-testid="stHeader"] * {{
        color: var(--text-primary) !important;
    }}

    /* Pastikan SEMUA teks standar di canvas & markdown mengikuti tema dengan kontras tinggi */
    .stApp p,
    .stApp label,
    .stApp li,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span {{
        color: var(--text-primary) !important;
    }}

    .stApp span:not(.badge-status):not(.hero-chip):not(.status-dot):not(.step-num) {{
        color: var(--text-primary) !important;
    }}

    .stApp [data-testid="stCaptionContainer"] p,
    .stApp small {{
        color: var(--text-secondary) !important;
    }}

    /* ==========================================================================
       SIDEBAR STYLING LENGKAP & KONTRAS TINGGI
       ========================================================================== */
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1.5px solid var(--border-color) !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {{
        color: var(--text-primary) !important;
    }}
    section[data-testid="stSidebar"] span:not(.badge-status) {{
        color: var(--text-primary) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: var(--text-secondary) !important;
    }}

    /* ==========================================================================
       CARD NAVIGATION BAR (TAB LIST UTAMA JADI CARD BESAR & MEWAH TANPA RED LINE)
       ========================================================================== */
    div[data-testid="stTabs"] [role="tablist"],
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
        background-color: var(--bg-card) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 8px !important;
        gap: 12px !important;
        margin-top: 8px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.06) !important;
        display: flex !important;
        width: 100% !important;
        flex-wrap: wrap !important;
    }}

    /* HAPUS SEMUA UNDERLINE / HIGHLIGHT MERAH BAWAAN STREAMLIT */
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"],
    div[data-testid="stTabs"] [role="tablist"] div[aria-hidden="true"],
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] div[data-baseweb="tab-border"] {{
        display: none !important;
        height: 0px !important;
        opacity: 0 !important;
        visibility: hidden !important;
        border: none !important;
        outline: none !important;
    }}

    /* Setiap Tombol Tab di dalam Card */
    div[data-testid="stTabs"] button[role="tab"],
    div[data-testid="stTabs"] button[data-baseweb="tab"],
    div[data-testid="stTabs"] [role="tablist"] button,
    div[data-testid="stTabs"] [data-baseweb="tab-list"] button {{
        flex: 1 1 200px !important;
        background-color: var(--tab-bg-inactive) !important;
        border: 1.5px solid var(--tab-border-inactive) !important;
        border-radius: 12px !important;
        padding: 14px 20px !important;
        font-size: 0.96rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
        white-space: nowrap !important;
        min-height: 52px !important;
        border-bottom-width: 1.5px !important;
    }}

    /* Pastikan SEMUA teks & ikon di dalam Tab INAKTIF SELALU kontras */
    div[data-testid="stTabs"] button[role="tab"] *,
    div[data-testid="stTabs"] button[data-baseweb="tab"] *,
    div[data-testid="stTabs"] button[role="tab"] p,
    div[data-testid="stTabs"] button[data-baseweb="tab"] p,
    div[data-testid="stTabs"] button[role="tab"] span,
    div[data-testid="stTabs"] button[data-baseweb="tab"] span {{
        color: var(--tab-text-inactive) !important;
        font-weight: 700 !important;
    }}

    /* Hover Tab Card */
    div[data-testid="stTabs"] button[role="tab"]:hover,
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {{
        border-color: #2563EB !important;
        background-color: var(--bg-card-hover) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.2) !important;
    }}
    div[data-testid="stTabs"] button[role="tab"]:hover * {{
        color: #2563EB !important;
    }}

    /* Active Tab Card: Gradien Biru Mewah dengan Glow */
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
    div[data-testid="stTabs"] [role="tablist"] button[aria-selected="true"],
    div[data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="true"] {{
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
        border: 1.5px solid #3B82F6 !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
        transform: translateY(-1px) !important;
    }}

    /* Teks Tab Aktif WAJIB Putih */
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] *,
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] *,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p {{
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }}

    /* ==========================================================================
       RADIO BUTTON SELECTORS (METODE PINDAI & THEME TOGGLE)
       ========================================================================== */
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        display: grid !important;
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 10px !important;
        margin-top: 10px !important;
        margin-bottom: 16px !important;
    }}

    div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
        background-color: var(--bg-card) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 12px 14px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
    }}

    /* Pastikan SEMUA teks label radio button SELALU terlihat jelas */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label *,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label p,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label span,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label div {{
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{
        border-color: #2563EB !important;
        background-color: var(--bg-card-hover) !important;
        transform: translateY(-1px) !important;
    }}

    /* Radio yang Terpilih */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {{
        border: 2px solid #2563EB !important;
        background-color: rgba(37, 99, 235, 0.1) !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.18) !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] *,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) * {{
        color: #2563EB !important;
        font-weight: 700 !important;
    }}

    /* Radio Toggle Tema Sidebar */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {{
        display: flex !important;
        grid-template-columns: none !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
        flex: 1 1 0 !important;
        padding: 8px 10px !important;
        text-align: center !important;
    }}

    /* ==========================================================================
       HERO BANNER (SELALU GRADASI BIRU DENGAN TEKS PUTIH)
       ========================================================================== */
    .hero-container {{
        background: linear-gradient(135deg, #091e42 0%, #172554 45%, #1d4ed8 100%);
        padding: 22px 28px;
        border-radius: 16px;
        color: #ffffff !important;
        margin-bottom: 18px;
        box-shadow: 0 10px 25px -5px rgba(29, 78, 216, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }}
    .hero-container * {{
        color: #ffffff !important;
    }}
    .hero-title {{
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff !important;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .hero-subtitle {{
        font-size: 0.92rem;
        color: #bfdbfe !important;
        margin-top: 4px;
        margin-bottom: 0px;
        font-weight: 400;
    }}
    .hero-chips {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 14px;
    }}
    .hero-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.14);
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #eff6ff !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}

    /* ==========================================================================
       STEPPER WORKFLOW VISUAL
       ========================================================================== */
    .stepper-box {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: var(--bg-card) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 14px;
        padding: 12px 18px;
        margin-bottom: 20px;
        gap: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        color: var(--text-primary) !important;
    }}
    .step-card {{
        display: flex;
        align-items: center;
        gap: 10px;
        flex: 1;
        padding: 8px 12px;
        border-radius: 10px;
        transition: all 0.2s ease;
    }}
    .step-card * {{
        color: var(--text-primary) !important;
    }}
    .step-card.active {{
        background: rgba(37, 99, 235, 0.08) !important;
        border: 1.5px solid #2563eb !important;
        box-shadow: 0 0 12px rgba(37, 99, 235, 0.15);
    }}
    .step-card.completed {{
        background: rgba(16, 185, 129, 0.08) !important;
        border: 1.5px solid #10b981 !important;
    }}
    .step-card.idle {{
        opacity: 0.65;
        border: 1px solid transparent;
    }}
    .step-num {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.92rem;
        flex-shrink: 0;
    }}
    .step-card.active .step-num {{
        background: #2563eb;
        color: #ffffff !important;
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.4);
    }}
    .step-card.completed .step-num {{
        background: #10b981;
        color: #ffffff !important;
    }}
    .step-card.idle .step-num {{
        background: var(--tab-bg-inactive);
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-color);
    }}
    .step-label {{
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: var(--text-secondary) !important;
    }}
    .step-title {{
        font-size: 0.88rem;
        font-weight: 700;
        line-height: 1.2;
        color: var(--text-primary) !important;
    }}
    .step-arrow {{
        color: var(--text-muted) !important;
        font-weight: bold;
        font-size: 1.1rem;
        user-select: none;
    }}

    /* ==========================================================================
       METRIC CARDS, CARDS & EMPTY STATE
       ========================================================================== */
    div[data-testid="stMetric"] {{
        background-color: var(--bg-card) !important;
        padding: 14px 18px !important;
        border-radius: 14px !important;
        border: 1.5px solid var(--border-color) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.15) !important;
        border-color: #2563EB !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] span {{
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] div {{
        color: var(--text-primary) !important;
        font-weight: 800 !important;
    }}

    .empty-state-card {{
        background-color: var(--bg-card) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 16px;
        padding: 36px 20px;
        text-align: center;
        margin-bottom: 16px;
        color: var(--text-primary) !important;
    }}
    .empty-title {{
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 4px;
        color: var(--text-primary) !important;
    }}
    .empty-sub {{
        font-size: 0.88rem;
        color: var(--text-secondary) !important;
        max-width: 440px;
        margin: 0 auto 16px auto;
        line-height: 1.5;
    }}
    .empty-pills {{
        display: flex;
        gap: 8px;
        justify-content: center;
        flex-wrap: wrap;
    }}
    .pill-mini {{
        background-color: var(--tab-bg-inactive) !important;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 600;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}

    .ticket-card {{
        background-color: var(--bg-card) !important;
        border: 2px solid #2563eb !important;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.15);
        position: relative;
        overflow: hidden;
        color: var(--text-primary) !important;
    }}
    .ticket-card * {{
        color: var(--text-primary) !important;
    }}
    .ticket-card::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #2563eb, #38bdf8, #10b981);
    }}
    .ticket-card hr {{
        border-top: 1px solid var(--border-color) !important;
    }}

    .ticket-card-invalid {{
        background-color: var(--bg-card) !important;
        border: 2px solid #ef4444 !important;
        border-radius: 16px;
        padding: 26px 20px;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.15);
        color: var(--text-primary) !important;
    }}
    .ticket-card-invalid * {{
        color: var(--text-primary) !important;
    }}

    /* Authentic Indonesian License Plate */
    .plate-badge-id {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(180deg, #111827 0%, #030712 100%);
        color: #ffffff !important;
        font-family: 'Courier New', 'Consolas', monospace;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 4px;
        padding: 10px 24px;
        border-radius: 10px;
        border: 3px solid #334155;
        box-shadow: inset 0 2px 4px rgba(255,255,255,0.15), 0 6px 16px rgba(0,0,0,0.45);
        text-shadow: 0 2px 4px rgba(0,0,0,0.7);
        margin: 8px 0 12px 0;
    }}

    /* Badges Status */
    .badge-status {{
        display: inline-block;
        padding: 7px 16px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: 0.2px;
    }}
    .status-dosen {{ background: rgba(16, 185, 129, 0.18); color: #10B981 !important; border: 1.5px solid rgba(16, 185, 129, 0.4); }}
    .status-mhs {{ background: rgba(37, 99, 235, 0.18); color: #2563EB !important; border: 1.5px solid rgba(37, 99, 235, 0.4); }}
    .status-tamu {{ background: rgba(245, 158, 11, 0.18); color: #D97706 !important; border: 1.5px solid rgba(245, 158, 11, 0.4); }}
    .status-invalid {{ background: rgba(239, 68, 68, 0.18); color: #DC2626 !important; border: 1.5px solid rgba(239, 68, 68, 0.4); }}
    .status-uncertain {{ background: rgba(234, 179, 8, 0.18); color: #CA8A04 !important; border: 1.5px solid rgba(234, 179, 8, 0.4); }}

    /* Action Buttons */
    div.stButton > button {{
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 12px 20px !important;
        transition: all 0.2s ease !important;
    }}

    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"],
    div.stButton > button[kind="primary"] *,
    div.stButton > button[data-testid="baseButton-primary"] * {{
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
        border: 1px solid #3B82F6 !important;
        color: white !important;
        font-weight: 800 !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35) !important;
    }}

    div.stButton > button[kind="primary"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.5) !important;
        border-color: #60A5FA !important;
    }}

    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="baseButton-secondary"] {{
        background: var(--bg-card) !important;
        border: 1.5px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }}
    div.stButton > button[kind="secondary"] * {{
        color: var(--text-primary) !important;
    }}

    div.stButton > button[kind="secondary"]:hover {{
        border-color: #2563EB !important;
        transform: translateY(-1px) !important;
    }}

    /* Alerts / Callout Styling */
    div[data-testid="stAlert"] {{
        border-radius: 14px !important;
        border-width: 1.5px !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }}
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentInfo"]) {{
        background-color: var(--alert-info-bg) !important;
        border: 1.5px solid var(--alert-info-border) !important;
    }}
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentInfo"]) * {{
        color: var(--alert-info-text) !important;
        font-weight: 500 !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stNotificationContentSuccess"]) {{
        background-color: var(--alert-success-bg) !important;
        border: 1.5px solid var(--alert-success-border) !important;
    }}
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentSuccess"]) * {{
        color: var(--alert-success-text) !important;
        font-weight: 500 !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]) {{
        background-color: var(--alert-warning-bg) !important;
        border: 1.5px solid var(--alert-warning-border) !important;
    }}
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]) * {{
        color: var(--alert-warning-text) !important;
        font-weight: 500 !important;
    }}

    /* Forms, DataFrame & Expanders */
    div[data-testid="stForm"] {{
        background-color: var(--bg-card) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 22px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04) !important;
    }}

    div[data-testid="stDataFrame"] {{
        background-color: var(--bg-card) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04) !important;
    }}

    div[data-testid="stExpander"] {{
        background-color: var(--bg-card) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 14px !important;
    }}
    div[data-testid="stExpander"] summary span {{
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }}

    /* Input & Select Controls */
    div[data-baseweb="input"] > div,
    input, textarea {{
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        border-color: var(--input-border) !important;
        border-radius: 10px !important;
    }}
    input::placeholder, textarea::placeholder {{
        color: var(--text-muted) !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        border-color: var(--input-border) !important;
        border-radius: 10px !important;
    }}
    div[data-baseweb="select"] * {{
        color: var(--input-text) !important;
    }}

    /* Selectbox Dropdown Menu Popover */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {{
        background-color: var(--bg-card) !important;
        border: 1.5px solid var(--border-color) !important;
    }}
    li[role="option"] {{
        color: var(--text-primary) !important;
        background-color: var(--bg-card) !important;
    }}
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {{
        background-color: rgba(37, 99, 235, 0.1) !important;
        color: #2563EB !important;
    }}

    /* Responsive adjustments */
    @media (max-width: 768px) {{
        div[data-testid="stTabs"] [role="tablist"] {{
            flex-direction: column !important;
        }}
        div[data-testid="stTabs"] button[role="tab"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
        }}
        div[data-testid="stRadio"] > div[role="radiogroup"] {{
            grid-template-columns: 1fr !important;
        }}
        .hero-title {{
            font-size: 1.35rem;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# Hitung data kapasitas real-time untuk header & sidebar
total_slots = 100
parked_now = db_manager.get_currently_parked_count()
available_slots = max(0, total_slots - parked_now)
pct_used = min(1.0, parked_now / total_slots)
is_model_available = os.path.isfile(DEFAULT_MODEL_PATH)

st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">🅿️ SMART PARKING KAMPUS</div>
    <div class="hero-subtitle">Sistem Otomasi Gerbang: Deteksi Plat Nomor (YOLO) + Pembacaan Karakter (EasyOCR) + Verifikasi Database Kampus</div>
    <div class="hero-chips">
        <span class="hero-chip">🟢 Sisa Slot: <b>{available_slots} / {total_slots}</b></span>
        <span class="hero-chip">🚗 Sedang Parkir: <b>{parked_now} Unit</b></span>
        <span class="hero-chip">{'✅ Model YOLO Aktif' if is_model_available else '⚠️ Model Belum Terpasang'}</span>
        <span class="hero-chip">⚡ Alur: Deteksi ➔ Verifikasi ➔ Buka Gerbang</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR: KONTROL OPERASIONAL GERBANG
# ==============================================================================
with st.sidebar:
    st.header("🏢 Pos Gerbang Parkir")
    
    gate_dir_mode = st.selectbox(
        "Mode Gerbang",
        [
            "🤖 Otomatis (Cerdas Masuk / Keluar)",
            "🟢 Khusus Pos Masuk (IN)",
            "🔴 Khusus Pos Keluar (OUT)",
        ],
        help="Mode Otomatis akan mengecek database: jika kendaraan belum di dalam kampus maka otomatis MASUK, jika sudah di dalam maka otomatis KELUAR dan hitung durasi."
    )

    if is_model_available:
        st.success("✅ Model YOLO Aktif")
    else:
        st.warning("⚠️ Model Belum Terpasang")

    st.divider()
    st.subheader("🅿️ Kapasitas Slot Parkir")
    st.progress(pct_used)
    st.write(f"Terisi: **{parked_now}** / {total_slots} Kendaraan")
    if available_slots > 15:
        st.success(f"🟢 Sisa: **{available_slots} Slot Tersedia**")
    elif available_slots > 0:
        st.warning(f"🟡 Hampir Penuh: Sisa **{available_slots} Slot**")
    else:
        st.error("🔴 PARKIR PENUH (0 Slot)")

    st.divider()
    st.subheader("💰 Tarif Parkir Resmi")
    st.markdown("""
    - **👨‍🏫 Dosen**: Gratis (Rp 0)
    - **🎓 Mahasiswa**: Rp 2.000 *(Flat Seharian)*
    - **🚗 Tamu / Umum**:
      - 1 Jam Pertama: **Rp 3.000**
      - Jam Berikutnya: **+Rp 2.000 / jam**
    """)

    st.divider()
    with st.expander("⚙️ Parameter Teknis & AI (Lanjutan)", expanded=False):
        st.caption("Pengaturan teknis ini sudah dioptimalkan secara default:")
        model_path_input = st.text_input("Path Model YOLO", value=DEFAULT_MODEL_PATH)
        is_model_available = os.path.isfile(model_path_input)
        
        conf_threshold = st.slider("Confidence YOLO", 0.10, 0.90, 0.30, 0.05)
        ocr_min_conf = st.slider("Confidence Min OCR", 0.15, 0.80, 0.25, 0.05)
        frame_skip = st.slider("Frekuensi Frame (Webcam)", 2, 16, 3, 1)
        cooldown_sec = st.number_input("Cooldown Duplikasi (detik)", 10, 600, 60, 10)

# ==============================================================================
# NAVIGASI UTAMA (4 TAB TERSTRUKTUR & JELAS)
# ==============================================================================
tab_gate, tab_vehicles, tab_history, tab_guide = st.tabs(
    [
        "🚪 Pos Gerbang (Pemeriksaan & Verifikasi)",
        "👥 Data Kendaraan Kampus",
        "📊 Laporan & Riwayat",
        "📘 Panduan & Status AI",
    ]
)

# Helper untuk memuat model YOLO dan EasyOCR
@st.cache_resource
def get_yolo_and_ocr(path: str):
    from ultralytics import YOLO
    import easyocr
    m = YOLO(path)
    r = easyocr.Reader(["en"], gpu=False)
    return m, r

@st.cache_resource
def get_ocr_only():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)

# ==============================================================================
# TAB 1: POS GERBANG (PEMERIKSAAN & VERIFIKASI)
# ==============================================================================
with tab_gate:
    # State untuk menyimpan plat terakhir yang dideteksi
    if "current_plate" not in st.session_state:
        st.session_state["current_plate"] = None
    if "current_conf" not in st.session_state:
        st.session_state["current_conf"] = 1.0
    if "current_source" not in st.session_state:
        st.session_state["current_source"] = "Belum Ada"

    active_plate = st.session_state.get("current_plate")

    # Stepper Workflow Visual (Langkah 1 -> Langkah 2 -> Langkah 3)
    if not active_plate:
        s1_class = "active"
        s2_class = "idle"
        s3_class = "idle"
        s1_icon = "📸"
        s2_icon = "⏳"
        s3_icon = "🚪"
        s1_status = "Sedang Berlangsung"
        s2_status = "Menunggu Plat"
        s3_status = "Menunggu Verifikasi"
    elif active_plate == "INVALID_PLATE":
        s1_class = "active"
        s2_class = "idle"
        s3_class = "idle"
        s1_icon = "❌"
        s2_icon = "🚫"
        s3_icon = "🔒"
        s1_status = "Bukan Plat Kendaraan"
        s2_status = "Verifikasi Ditolak"
        s3_status = "Gerbang Terkunci"
    else:
        s1_class = "completed"
        s2_class = "completed"
        s3_class = "active"
        s1_icon = "✅"
        s2_icon = "✅"
        s3_icon = "🔓"
        s1_status = "Plat Terdeteksi"
        s2_status = "Terverifikasi"
        s3_status = "Siap Buka Gerbang"

    st.markdown(f"""
    <div class="stepper-box">
        <div class="step-card {s1_class}">
            <div class="step-num">{s1_icon}</div>
            <div class="step-meta">
                <span class="step-label">LANGKAH 1</span>
                <span class="step-title">Pindai / Masukkan Plat</span>
            </div>
        </div>
        <div class="step-arrow">➔</div>
        <div class="step-card {s2_class}">
            <div class="step-num">{s2_icon}</div>
            <div class="step-meta">
                <span class="step-label">LANGKAH 2</span>
                <span class="step-title">Verifikasi Hak & Tarif</span>
            </div>
        </div>
        <div class="step-arrow">➔</div>
        <div class="step-card {s3_class}">
            <div class="step-num">{s3_icon}</div>
            <div class="step-meta">
                <span class="step-label">LANGKAH 3</span>
                <span class="step-title">Buka Palang Gerbang</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_verify = st.columns([1.1, 1.25], gap="medium")

    # --------------------------------------------------------------------------
    # KOLOM KIRI: LANGKAH 1 — INPUT & DETEKSI PLAT
    # --------------------------------------------------------------------------
    with col_input:
        st.markdown("#### 📸 Langkah 1: Pindai Plat Kendaraan")
        
        mode_input = st.radio(
            "Pilih Metode Deteksi:",
            [
                "⚡ Kamera Cepat (Snapshot 1-Klik)",
                "⌨️ Ketik Plat Manual",
                "📹 Kamera Live (Streaming WebRTC)",
            ],
            horizontal=False,
            label_visibility="collapsed",
        )

        # METODE 1: KAMERA SNAPSHOT (NATIVE BROWSER HTML5 - 60 FPS, 0% LAG)
        if mode_input == "⚡ Kamera Cepat (Snapshot 1-Klik)":
            if not is_model_available:
                st.warning("⚠️ File model YOLO belum ditemukan. Gunakan mode 'Ketik Plat Manual' atau pasang model YOLO.")
            else:
                st.info("💡 **Arahkan kamera ke plat kendaraan** (bisa posisi depan maupun serong samping kanan/kiri), lalu klik tombol ambil foto di bawah. Sangat cepat, tajam, dan anti-lag!")
                cam_shot = st.camera_input("Arahkan kamera ke plat nomor kendaraan:", key="native_cam_input")
                if cam_shot is not None:
                    file_bytes = np.asarray(bytearray(cam_shot.read()), dtype=np.uint8)
                    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    with st.spinner("AI sedang membaca plat nomor dari foto..."):
                        from ocr_engine import extract_plate_text
                        model, reader = get_yolo_and_ocr(DEFAULT_MODEL_PATH)
                        eff_conf = min(conf_threshold, 0.25)
                        preds = model.predict(source=img_cv, conf=eff_conf, verbose=False, imgsz=640)
                        ih, iw = img_cv.shape[:2]
                        candidates = extract_candidate_plate_boxes(preds[0].boxes if preds else [], frame_w=iw, frame_h=ih)
                        found = []
                        for box, box_conf, kind in candidates:
                            x1, y1, x2, y2 = box
                            crop = img_cv[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
                            if crop.size > 0:
                                t, c, _ = extract_plate_text(reader, crop)
                                if t:
                                    cleaned = disambiguate_plate(t)
                                    if is_valid_plate_pattern(cleaned):
                                        found.append((cleaned, c))

                        if found:
                            best_t, best_c = max(found, key=lambda x: x[1])
                            st.session_state["current_plate"] = best_t
                            st.session_state["current_conf"] = best_c
                            st.session_state["current_source"] = "Kamera Cepat"
                            st.toast(f"Plat terdeteksi: {best_t}", icon="✅")
                            st.rerun()
                        else:
                            st.session_state["current_plate"] = "INVALID_PLATE"
                            st.session_state["current_conf"] = 0.0
                            st.session_state["current_source"] = "Bukan Kendaraan"
                            st.rerun()

        # METODE 2: KETIK MANUAL
        elif mode_input == "⌨️ Ketik Plat Manual":
            st.info("💡 Gunakan jika plat nomor kotor/rusak dan tidak terbaca oleh AI:")
            manual_in = st.text_input("Ketik Nomor Plat Kendaraan", placeholder="Contoh: B 6802 JPN / F 4526 FCF")
            if st.button("🔎 Periksa Nomor Plat", type="primary", use_container_width=True):
                cleaned_man = disambiguate_plate(manual_in.strip())
                if not manual_in.strip():
                    st.warning("Masukkan nomor plat terlebih dahulu.")
                elif not is_valid_plate_pattern(cleaned_man):
                    st.error("❌ Format plat tidak valid. Masukkan format plat Indonesia yang benar (Contoh: B 1234 XYZ, F 4526 FCF). Jangan masukkan kata acak atau nama.")
                else:
                    st.session_state["current_plate"] = cleaned_man
                    st.session_state["current_conf"] = 1.0
                    st.session_state["current_source"] = "Input Manual"
                    st.rerun()

        # METODE 4: KAMERA STREAMING WEBRTC
        else:
            try:
                from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
                from ocr_engine import extract_plate_text
                model, reader = get_yolo_and_ocr(DEFAULT_MODEL_PATH)

                class LiveCameraProcessor(VideoProcessorBase):
                    def __init__(self):
                        self.lock = threading.Lock()
                        self.latest_frame = None
                        self.frame_count = 0
                        self.tracker = SmoothBoxTracker(alpha=0.65, max_miss_frames=8)

                    def recv(self, frame):
                        img = frame.to_ndarray(format="bgr24")
                        self.frame_count += 1
                        h, w = img.shape[:2]

                        with self.lock:
                            self.latest_frame = img.copy()

                        # Deteksi kotak plat dengan YOLO tiap beberapa frame
                        step = max(2, min(frame_skip, 4))
                        if self.frame_count % step == 0:
                            cands = []
                            try:
                                preds = model.predict(source=img, conf=0.25, verbose=False, imgsz=480)
                                if preds and len(preds[0].boxes) > 0:
                                    cands = extract_candidate_plate_boxes(preds[0].boxes, frame_w=w, frame_h=h)
                            except Exception:
                                cands = []
                            with self.lock:
                                cur_box = self.tracker.update(cands)
                        else:
                            with self.lock:
                                cur_box = self.tracker.get_box()

                        if cur_box:
                            x1, y1, x2, y2 = cur_box
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            # Kotak hijau neon dinamis yang presisi & stabil mengikuti plat nomor
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 240, 0), 3)

                            # Badge status "PLAT TERDETEKSI" di atas kotak
                            badge_text = "PLAT TERDETEKSI (KLIK SCAN)"
                            t_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
                            bg_y1 = max(0, y1 - 28)
                            bg_y2 = y1
                            cv2.rectangle(img, (x1, bg_y1), (min(w, x1 + t_size[0] + 16), bg_y2), (0, 200, 0), -1)
                            cv2.putText(img, badge_text, (x1 + 8, bg_y1 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                        else:
                            # Bila plat belum terarah: Sudut bidik kamera tipis (viewfinder brackets)
                            cx, cy = w // 2, h // 2
                            vw = min(int(w * 0.85), 340)
                            vh = min(int(h * 0.35), 140)
                            vx1, vy1 = cx - vw // 2, cy - vh // 2
                            vx2, vy2 = cx + vw // 2, cy + vh // 2
                            c_len = 22
                            c_color = (220, 220, 220)

                            # 4 sudut bidik tipis
                            cv2.line(img, (vx1, vy1), (vx1 + c_len, vy1), c_color, 2)
                            cv2.line(img, (vx1, vy1), (vx1, vy1 + c_len), c_color, 2)
                            cv2.line(img, (vx2, vy1), (vx2 - c_len, vy1), c_color, 2)
                            cv2.line(img, (vx2, vy1), (vx2, vy1 + c_len), c_color, 2)
                            cv2.line(img, (vx1, vy2), (vx1 + c_len, vy2), c_color, 2)
                            cv2.line(img, (vx1, vy2), (vx1, vy2 - c_len), c_color, 2)
                            cv2.line(img, (vx2, vy2), (vx2 - c_len, vy2), c_color, 2)
                            cv2.line(img, (vx2, vy2), (vx2, vy2 - c_len), c_color, 2)

                            guide_lbl = "Arahkan plat (depan / serong kanan / kiri)"
                            g_size = cv2.getTextSize(guide_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                            cv2.putText(img, guide_lbl, (cx - g_size[0] // 2, max(22, vy1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c_color, 1, cv2.LINE_AA)

                        return type(frame).from_ndarray(img, format="bgr24")

                st.info("💡 **Dukungan Multi-Sudut Pos Gerbang**: Kamera dapat diarahkan lurus dari depan maupun **serong dari samping kanan/kiri** (hingga sudut 45°). AI otomatis meluruskan kemiringan plat!")
                ctx = webrtc_streamer(
                    key="gate-camera-stream",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=LiveCameraProcessor,
                    media_stream_constraints={
                        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 30}},
                        "audio": False
                    },
                    async_processing=True,
                )

                btn_scan = st.button("📸 SCAN / AMBIL PLAT SEKARANG", type="primary", use_container_width=True)
                if btn_scan and ctx.video_processor:
                    with ctx.video_processor.lock:
                        snap = ctx.video_processor.latest_frame
                        last_tracked = ctx.video_processor.tracker.get_box()
                    if snap is not None:
                        with st.spinner("AI sedang memeriksa plat nomor kendaraan..."):
                            sh, sw = snap.shape[:2]
                            eff_conf = min(conf_threshold, 0.25)
                            preds = model.predict(source=snap, conf=eff_conf, verbose=False, imgsz=640)
                            candidates = extract_candidate_plate_boxes(preds[0].boxes if preds else [], frame_w=sw, frame_h=sh)
                            if not candidates and last_tracked:
                                candidates = [(last_tracked, 0.50, "tracked_fallback")]
                            found = []
                            for box, box_conf, kind in candidates:
                                x1, y1, x2, y2 = box
                                crop = snap[max(0, y1):min(sh, y2), max(0, x1):min(sw, x2)]
                                if crop.size > 0:
                                    t, c, _ = extract_plate_text(reader, crop)
                                    if t:
                                        cleaned = disambiguate_plate(t)
                                        if is_valid_plate_pattern(cleaned):
                                            found.append((cleaned, c))

                            if found:
                                best_t, best_c = max(found, key=lambda x: x[1])
                                st.session_state["current_plate"] = best_t
                                st.session_state["current_conf"] = best_c
                                st.session_state["current_source"] = "Kamera Streaming"
                                st.toast(f"Plat terdeteksi: {best_t}", icon="✅")
                                st.rerun()
                            else:
                                st.session_state["current_plate"] = "INVALID_PLATE"
                                st.session_state["current_conf"] = 0.0
                                st.session_state["current_source"] = "Bukan Kendaraan"
                                st.rerun()
            except Exception as e:
                st.error(f"Gagal mengaktifkan kamera streaming: {e}")

    # --------------------------------------------------------------------------
    # KOLOM KANAN: LANGKAH 2 & 3 — HASIL VERIFIKASI & AKSI GERBANG
    # --------------------------------------------------------------------------
    with col_verify:
        st.markdown("#### 🎫 Langkah 2 & 3: Verifikasi & Aksi Gerbang")

        if not active_plate:
            # EMPTY STATE (MENUNGGU PEMERIKSAAN)
            st.markdown(f"""
            <div class="empty-state-card">
                <div style="font-size: 2.8rem; margin-bottom: 6px;">🚗 🔍</div>
                <div class="empty-title">Menunggu Kendaraan di Pos Gerbang</div>
                <div class="empty-sub">
                    Gunakan kamera snapshot 1-klik atau ketik plat di sebelah kiri.<br>
                    Sistem AI akan otomatis membaca plat nomor, memverifikasi status kampus, dan menentukan tarif parkir.
                </div>
                <div class="empty-pills">
                    <span class="pill-mini">🤖 Verifikasi Database Otomatis</span>
                    <span class="pill-mini">🎓 Dosen: Rp0</span>
                    <span class="pill-mini">🎒 Mahasiswa: Rp2.000 Flat</span>
                    <span class="pill-mini">🏷️ Tamu: Rp3.000 (Jam 1) + Rp2.000/Jam</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif active_plate == "INVALID_PLATE":
            # REJECTION STATE (BUKAN KENDARAAN)
            st.markdown("""
            <div class="ticket-card-invalid">
                <div style="font-size: 2.5rem; margin-bottom: 6px;">🚫</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #dc2626;">BUKAN PLAT KENDARAAN</div>
                <div style="font-size: 0.92rem; color: #64748b; margin-top: 6px; line-height: 1.5;">
                    AI mendeteksi objek ini <b>bukan plat nomor kendaraan yang sah</b> (wajah, pakaian, teks acak, atau benda lain).<br>
                    Palang gerbang <b>tetap terkunci</b> demi keamanan kampus.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Pindai Ulang Kendaraan", type="secondary", use_container_width=True):
                st.session_state["current_plate"] = None
                st.rerun()

        else:
            # TICKET STATE (PLAT VALID)
            cleaned_plat = disambiguate_plate(active_plate)
            vehicle_record = db_manager.find_vehicle(cleaned_plat)
            active_entry = db_manager.get_active_entry(cleaned_plat)

            # Tentukan arah gerbang secara cerdas
            if "Otomatis" in gate_dir_mode:
                if active_entry:
                    gate_code = "OUT"
                    gate_status_label = "🔴 KELUAR (OUT) — 🤖 Terdeteksi Sedang Parkir"
                else:
                    gate_code = "IN"
                    gate_status_label = "🟢 MASUK (IN) — 🤖 Terdeteksi Baru Tiba"
            elif "IN" in gate_dir_mode:
                gate_code = "IN"
                gate_status_label = "🟢 MASUK (IN) — Mode Gerbang Masuk"
            else:
                gate_code = "OUT"
                gate_status_label = "🔴 KELUAR (OUT) — Mode Gerbang Keluar"

            entry_record = active_entry if gate_code == "OUT" else None
            dec_result = decide(
                cleaned_plat,
                confidence=st.session_state.get("current_conf", 1.0),
                vehicle_data=vehicle_record,
                min_conf=ocr_min_conf,
                gate_direction=gate_code,
                entry_record=entry_record,
            )

            # Tentukan warna badge status
            if dec_result["status_code"] == "REGISTERED":
                if dec_result["category"] == "Dosen":
                    badge_style = "status-dosen"
                    status_title = "🎓 DOSEN TERDAFTAR — PARKIR GRATIS"
                else:
                    badge_style = "status-mhs"
                    status_title = f"🎒 {dec_result['category'].upper()} TERDAFTAR — WAJIB BAYAR"
            elif dec_result["status_code"] == "OCR_UNCERTAIN":
                badge_style = "status-uncertain"
                status_title = "⚠️ TEKS BURAM / OCR KURANG YAKIN"
            else:
                badge_style = "status-tamu"
                status_title = "🏷️ KENDARAAN TAMU (LUAR KAMPUS) — WAJIB BAYAR"

            # Info durasi & waktu parkir
            if gate_code == "OUT":
                dur_val = dec_result.get("duration_display", "-")
                entry_str = dec_result.get("entry_time")
                entry_disp = entry_str.replace("T", " ") if entry_str else "Tidak Tercatat (Langsung Keluar)"
                extra_gate_info = (
                    f"<div>⏱️ <b>Durasi Parkir:</b><br><span style='font-size:1.15rem; font-weight:700; color:#059669;'>{dur_val}</span></div>"
                    f"<div>🕒 <b>Waktu Masuk:</b><br><span style='font-size:0.92rem; color:var(--text-secondary);'>{entry_disp}</span></div>"
                )
            else:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                extra_gate_info = (
                    "<div>⏱️ <b>Status Sesi:</b><br><span style='font-size:1.1rem; font-weight:700; color:#2563eb;'>Tiket Baru Masuk</span></div>"
                    f"<div>🕒 <b>Waktu Masuk:</b><br><span style='font-size:0.92rem; color:var(--text-secondary);'>{now_str}</span></div>"
                )

            dur_mins = dec_result.get("duration_minutes", 0)
            fee_note = ""
            if "Tamu" in str(dec_result.get("category", "")):
                if gate_code == "IN":
                    fee_note = "<div style='font-size:0.75rem; color:var(--text-secondary);'>Tarif 1 Jam Pertama (+Rp2.000/jam)</div>"
                elif dur_mins > 60:
                    extra_h = (dur_mins - 1) // 60
                    fee_note = f"<div style='font-size:0.75rem; color:var(--text-secondary);'>1 Jam Pertama (Rp3.000) + {extra_h} Jam x Rp2.000</div>"
                else:
                    fee_note = "<div style='font-size:0.75rem; color:var(--text-secondary);'>Tarif 1 Jam Pertama (<= 60 Menit)</div>"
            elif "Mahasiswa" in str(dec_result.get("category", "")):
                fee_note = "<div style='font-size:0.75rem; color:var(--text-secondary);'>Tarif Flat Subsidi Kampus (Seharian)</div>"
            elif "Dosen" in str(dec_result.get("category", "")):
                fee_note = "<div style='font-size:0.75rem; color:#10b981;'>Akses Sivitas Akademika Gratis</div>"

            # Kartu Tiket Digital Modern
            card_html = f"""
            <div class="ticket-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px;">
                    <div>
                        <div style="font-size:0.75rem; color:var(--text-secondary); font-weight:700; letter-spacing:0.5px; text-transform:uppercase;">Plat Nomor Terbaca:</div>
                        <div class="plate-badge-id">{format_plate_display(cleaned_plat)}</div>
                        <div style="font-size:0.82rem; color:var(--text-secondary);">Sumber: <b>{st.session_state.get('current_source')}</b> (OCR Conf: {st.session_state.get('current_conf', 1.0):.0%})</div>
                    </div>
                    <div>
                        <span class="badge-status {badge_style}">{status_title}</span>
                    </div>
                </div>
                <hr style="margin: 16px 0; border: none; border-top: 1px solid rgba(128,128,128,0.2);" />
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 14px; font-size: 0.95rem;">
                    <div>👤 <b>Nama Pemilik:</b><br><span style="font-size:1.1rem; font-weight:700;">{dec_result.get('owner_name') or 'Tidak Valid'}</span></div>
                    <div>🏷️ <b>Kategori Civitas:</b><br><span style="font-size:1.1rem; font-weight:700;">{dec_result.get('category')}</span></div>
                    <div>💵 <b>Total Biaya Parkir:</b><br><span style="font-size:1.45rem; font-weight:800; color:#2563eb;">{rupiah(dec_result.get('fee'))}</span>{fee_note}</div>
                    <div>🚪 <b>Aksi Gerbang:</b><br><span style="font-size:1.02rem; font-weight:700;">{gate_status_label}</span></div>
                    {extra_gate_info}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # Cek Peringatan Duplikasi Transaksi
            recent = db_manager.check_recent_transaction(cleaned_plat, cooldown_seconds=cooldown_sec, gate_direction=gate_code)
            if recent:
                st.warning(f"⚠️ Peringatan: Kendaraan {format_plate_display(cleaned_plat)} sudah tercatat pada {recent['created_at']} (arah {gate_code}).")

            # Expander Opsional untuk Koreksi Plat & Catatan Petugas (Hanya jika diperlukan)
            with st.expander("✏️ Koreksi Plat / Catatan Tambahan (Klik jika plat salah baca)", expanded=False):
                st.caption("Ubah plat di bawah ini hanya jika AI salah membaca karakter plat nomor:")
                koreksi_plat = st.text_input("Koreksi Plat Nomor", value=format_plate_display(cleaned_plat), key="input_koreksi_plat")
                catatan_petugas = st.text_input("Catatan Petugas (Opsional)", placeholder="Contoh: Mengantar dokumen ke Rektorat", key="input_catatan_petugas")
                if st.button("Terapkan Koreksi Plat", type="secondary"):
                    if koreksi_plat.strip():
                        st.session_state["current_plate"] = koreksi_plat.strip()
                        st.session_state["current_source"] = "Koreksi Manual"
                        st.rerun()

            # LANGKAH 3: TOMBOL UTAMA BUKA GERBANG (1-KLIK LANGSUNG JALAN)
            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col1:
                if gate_code == "IN":
                    btn_open = st.button("🟢 BUKA GERBANG MASUK & CATAT TRANSAKSI", type="primary", use_container_width=True)
                else:
                    fee_txt = rupiah(dec_result.get("fee", 0))
                    btn_open = st.button(f"🔴 BUKA GERBANG KELUAR & SELESAI PARKIR ({fee_txt})", type="primary", use_container_width=True)

            with btn_col2:
                btn_cancel = st.button("🔄 Periksa Lain", type="secondary", use_container_width=True)

            if btn_cancel:
                st.session_state["current_plate"] = None
                st.rerun()

            if btn_open:
                final_p = cleaned_plat
                rec = db_manager.find_vehicle(final_p)
                active_ent = db_manager.get_active_entry(final_p)

                if "Otomatis" in gate_dir_mode:
                    final_gate_code = "OUT" if active_ent else "IN"
                elif "IN" in gate_dir_mode:
                    final_gate_code = "IN"
                else:
                    final_gate_code = "OUT"

                final_dec = decide(
                    final_p,
                    confidence=1.0,
                    vehicle_data=rec,
                    min_conf=ocr_min_conf,
                    gate_direction=final_gate_code,
                    entry_record=active_ent if final_gate_code == "OUT" else None,
                )

                catatan_val = st.session_state.get("input_catatan_petugas", "")

                tx_id = db_manager.save_transaction(
                    plate=final_p,
                    owner_name=final_dec.get("owner_name"),
                    category=final_dec.get("category"),
                    fee=final_dec.get("fee"),
                    decision=final_dec.get("decision"),
                    status="CONFIRMED",
                    gate_direction=final_gate_code,
                    notes=catatan_val,
                    duration_minutes=final_dec.get("duration_minutes", 0),
                    entry_time=final_dec.get("entry_time"),
                )
                st.success(f"🎉 Gerbang {final_gate_code} Terbuka! Transaksi #{tx_id} ({final_p}) berhasil dicatat.")
                st.session_state["current_plate"] = None
                st.rerun()

# ==============================================================================
# TAB 2: DATA KENDARAAN KAMPUS (ADMIN DATABASE & CRUD)
# ==============================================================================
with tab_vehicles:
    st.subheader("Data Master Kendaraan Sivitas Kampus")
    st.caption("Pusat pengelolaan data kendaraan resmi (Dosen & Mahasiswa) serta pemantauan status parkir aktif di lingkungan kampus.")

    # 1. Ringkasan Status Parkir Kendaraan Civitas
    all_reg_summary = db_manager.list_vehicles()
    total_civitas = len(all_reg_summary)
    inside_civitas = sum(1 for v in all_reg_summary if v.get("is_parked"))
    outside_civitas = total_civitas - inside_civitas

    c_sum1, c_sum2, c_sum3 = st.columns(3)
    c_sum1.metric("👥 Total Kendaraan Terdaftar", f"{total_civitas} Unit")
    c_sum2.metric("🟢 Sedang Parkir di Dalam", f"{inside_civitas} Unit")
    c_sum3.metric("⚪ Sedang di Luar Kampus", f"{outside_civitas} Unit")

    if inside_civitas > 0:
        st.success(f"🟢 **{inside_civitas} kendaraan sivitas terdaftar** saat ini sedang berada di dalam area parkir kampus.")

    # Sub-tabs CRUD: Daftar & Status, Tambah Baru, Edit Kendaraan, Hapus Kendaraan, Impor & Ekspor
    crud_tab1, crud_tab2, crud_tab3, crud_tab4, crud_tab5 = st.tabs([
        "📋 Daftar & Status Parkir",
        "➕ Tambah Kendaraan Baru",
        "✏️ Edit / Perbarui Data",
        "🗑️ Hapus Kendaraan",
        "📥 Impor & Ekspor CSV",
    ])

    # --------------------------------------------------------------------------
    # CRUD 1: DAFTAR & STATUS PARKIR (READ)
    # --------------------------------------------------------------------------
    with crud_tab1:
        f_c1, f_c2, f_c3 = st.columns([2, 1, 1])
        search_q = f_c1.text_input("🔍 Cari Plat / Nama Pemilik", "", key="search_veh_tab2")
        filter_k = f_c2.selectbox("Filter Kategori", ["Semua", "Dosen", "Mahasiswa"], key="filter_cat_tab2")
        filter_status = f_c3.selectbox(
            "Status Parkir",
            ["Semua Status", "🟢 Sedang di Dalam Saja", "⚪ Sedang di Luar Saja"],
            key="filter_park_tab2"
        )

        kendaraan_list = db_manager.list_vehicles(
            query=search_q,
            category_filter=filter_k if filter_k != "Semua" else None
        )

        if filter_status == "🟢 Sedang di Dalam Saja":
            kendaraan_list = [k for k in kendaraan_list if k.get("is_parked")]
        elif filter_status == "⚪ Sedang di Luar Saja":
            kendaraan_list = [k for k in kendaraan_list if not k.get("is_parked")]

        if kendaraan_list:
            display_rows = []
            for k in kendaraan_list:
                display_rows.append({
                    "Plat Nomor": format_plate_display(k["plate"]),
                    "Nama Pemilik": k.get("owner_name") or "-",
                    "Kategori": k.get("category"),
                    "Status Parkir": k.get("parking_status"),
                    "Waktu Masuk": k.get("entry_time").replace("T", " ") if k.get("entry_time") else "-",
                    "Terdaftar Sejak": (k.get("created_at") or "-")[:10],
                })
            df_display = pd.DataFrame(display_rows)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            csv_exp = df_display.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 Unduh Daftar Kendaraan (CSV)",
                data=csv_exp,
                file_name="data_kendaraan_kampus.csv",
                mime="text/csv",
                key="btn_download_veh_csv"
            )
        else:
            st.info("Belum ada kendaraan yang cocok dengan filter pencarian.")

    # --------------------------------------------------------------------------
    # CRUD 2: TAMBAH KENDARAAN BARU (CREATE)
    # --------------------------------------------------------------------------
    with crud_tab2:
        st.markdown("#### ➕ Registrasi Kendaraan Baru (Dosen / Mahasiswa)")
        st.caption("Masukkan plat nomor dan data sivitas kampus agar terverifikasi otomatis saat pemeriksaan kamera.")
        with st.form("form_tambah_kendaraan", clear_on_submit=True):
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                in_plat = st.text_input("Nomor Plat Kendaraan", placeholder="Contoh: B 1234 XYZ atau F 4526 FCF")
                in_kat = st.selectbox("Kategori Civitas", ["Mahasiswa", "Dosen"])
            with col_t2:
                in_nama = st.text_input("Nama Lengkap Pemilik", placeholder="Contoh: Farel Maulana / Prof. Dr. Hendra")
            
            submit_btn = st.form_submit_button("➕ Daftarkan Kendaraan", type="primary", use_container_width=True)
            if submit_btn:
                if not in_plat.strip() or not in_nama.strip():
                    st.error("Nomor plat dan nama pemilik wajib diisi!")
                else:
                    try:
                        is_up, msg = db_manager.upsert_vehicle(in_plat, in_nama, in_kat)
                        st.success(f"🎉 {msg}")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Gagal mendaftarkan kendaraan: {exc}")

    # --------------------------------------------------------------------------
    # CRUD 3: EDIT DATA KENDARAAN (UPDATE)
    # --------------------------------------------------------------------------
    with crud_tab3:
        st.markdown("#### ✏️ Perbarui / Koreksi Data Kendaraan")
        st.caption("Ubah nama pemilik, kategori, atau perbaiki nomor plat jika sebelumnya salah ketik.")

        if all_reg_summary:
            edit_options = {
                f"{format_plate_display(v['plate'])} — {v['owner_name']} ({v['category']})": v
                for v in all_reg_summary
            }
            selected_edit_label = st.selectbox(
                "Pilih Kendaraan yang Ingin Diperbarui:",
                ["— Pilih Kendaraan —"] + list(edit_options.keys()),
                key="select_edit_veh"
            )

            if selected_edit_label != "— Pilih Kendaraan —":
                chosen = edit_options[selected_edit_label]
                st.info(f"Status Kendaraan: **{chosen.get('parking_status')}**")

                with st.form("form_edit_kendaraan"):
                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        new_plat_input = st.text_input("Nomor Plat", value=format_plate_display(chosen["plate"]))
                        cur_idx = 0 if chosen.get("category") == "Dosen" else 1
                        new_kat_input = st.selectbox("Kategori", ["Dosen", "Mahasiswa"], index=cur_idx)
                    with col_e2:
                        new_nama_input = st.text_input("Nama Pemilik", value=chosen.get("owner_name") or "")

                    btn_save_edit = st.form_submit_button("💾 Simpan Perubahan Kendaraan", type="primary", use_container_width=True)
                    if btn_save_edit:
                        if not new_plat_input.strip() or not new_nama_input.strip():
                            st.error("Nomor plat dan nama pemilik tidak boleh kosong!")
                        else:
                            success = db_manager.update_vehicle(
                                old_plate=chosen["plate"],
                                new_owner_name=new_nama_input,
                                new_category=new_kat_input,
                                new_plate=new_plat_input,
                            )
                            if success:
                                st.success(f"✅ Data kendaraan {new_plat_input} berhasil diperbarui!")
                                st.rerun()
                            else:
                                st.error("Gagal memperbarui data kendaraan.")
        else:
            st.info("Belum ada kendaraan terdaftar untuk diedit.")

    # --------------------------------------------------------------------------
    # CRUD 4: HAPUS KENDARAAN (DELETE)
    # --------------------------------------------------------------------------
    with crud_tab4:
        st.markdown("#### 🗑️ Hapus Kendaraan Terdaftar")
        st.caption("Hapus data kendaraan dari daftar terdaftar jika pemilik sudah lulus atau tidak aktif lagi.")

        if all_reg_summary:
            del_options = {
                f"{format_plate_display(v['plate'])} — {v['owner_name']} ({v['category']})": v["plate"]
                for v in all_reg_summary
            }
            del_label = st.selectbox(
                "Pilih Kendaraan yang Ingin Dihapus:",
                ["— Pilih Kendaraan —"] + list(del_options.keys()),
                key="select_del_veh"
            )

            if del_label != "— Pilih Kendaraan —":
                target_plat = del_options[del_label]
                st.warning(f"⚠️ Perhatian: Anda akan menghapus data kendaraan **{format_plate_display(target_plat)}** dari database kampus.")
                confirm_del = st.checkbox(f"Saya yakin ingin menghapus data kendaraan {format_plate_display(target_plat)}", key="chk_del_veh")
                
                if st.button("🗑️ Hapus Kendaraan Sekarang", type="secondary", disabled=not confirm_del, use_container_width=True):
                    db_manager.delete_vehicle(target_plat)
                    st.success(f"Kendaraan {format_plate_display(target_plat)} berhasil dihapus dari database.")
                    st.rerun()
        else:
            st.info("Belum ada kendaraan terdaftar.")

    # --------------------------------------------------------------------------
    # CRUD 5: IMPOR & EKSPOR CSV
    # --------------------------------------------------------------------------
    with crud_tab5:
        st.markdown("#### 📥 Impor Data Kendaraan dari CSV")
        st.caption("Unggah file CSV dengan kolom: `plate, owner_name, category`")
        uploaded_csv = st.file_uploader("Pilih file CSV", type=["csv"], key="uploader_csv_v")

        if uploaded_csv is not None:
            try:
                df_csv = pd.read_csv(uploaded_csv, dtype=str).fillna("")
                req_cols = {"plate", "owner_name", "category"}
                if not req_cols.issubset(set(df_csv.columns)):
                    st.error("Kolom wajib CSV: plate, owner_name, category.")
                else:
                    st.dataframe(df_csv.head(5), use_container_width=True, hide_index=True)
                    if st.button("Proses Impor CSV", type="primary"):
                        count_success = 0
                        for _, row in df_csv.iterrows():
                            p = row["plate"].strip()
                            o = row["owner_name"].strip()
                            c = row["category"].strip().title()
                            if c in ["Dosen", "Mahasiswa"]:
                                db_manager.upsert_vehicle(p, o, c)
                                count_success += 1
                        st.success(f"{count_success} data berhasil diimpor.")
                        st.rerun()
            except Exception as e:
                st.error(f"Error membaca CSV: {e}")

        sample_csv = "plate,owner_name,category\nB 6802 JPN,Prof. Dr. Hendra,Dosen\nF 4526 FCF,Farel Maulana,Mahasiswa\n"
        st.download_button(
            "Unduh Template CSV",
            data=sample_csv,
            file_name="template_kendaraan_kampus.csv",
            mime="text/csv",
        )

# ==============================================================================
# TAB 3: LAPORAN & RIWAYAT TRANSAKSI (DASHBOARD & CRUD)
# ==============================================================================
with tab_history:
    st.subheader("Dashboard & Log Riwayat Transaksi")

    metrics = db_manager.get_dashboard_metrics()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Kendaraan Terdaftar", metrics["total_vehicles_registered"], help="Total kendaraan sivitas (Dosen & Mahasiswa) di database master kampus.")
    m2.metric("Sedang Parkir", f"{metrics['currently_parked']} Unit", help="Jumlah kendaraan yang saat ini masih berada di dalam area parkir kampus (sudah IN tapi belum OUT).")
    m3.metric("Total Pemeriksaan", metrics["total_examinations"], help="Total berapa kali palang gerbang dibuka / kendaraan diperiksa oleh AI (termasuk pemeriksaan gerbang Masuk dan Keluar).")
    m4.metric("Civitas Terdaftar", metrics["total_registered_inspected"], help="Berapa kali pemeriksaan melibatkan kendaraan sivitas kampus resmi (Dosen gratis Rp0, Mahasiswa flat Rp2.000).")
    m5.metric("Plat Tamu / Luar", metrics["total_unregistered_inspected"], help="Berapa kali pemeriksaan melibatkan pengunjung luar / belum terdaftar di database.")
    m6.metric("Total Pendapatan", rupiah(metrics["total_revenue"]), help="Akumulasi tarif parkir resmi yang berhasil dicatat sistem.")

    # PANDUAN PENJELASAN ISTILAH METRIK
    with st.expander("ℹ️ Penjelasan & Arti Metrik Dashboard (Klik untuk Membaca)", expanded=False):
        st.markdown("""
        - 🔍 **Total Pemeriksaan**: Total akumulasi berapa kali gerbang pos dibuka oleh sistem AI (baik saat kendaraan **Masuk / IN** maupun **Keluar / OUT**). Contoh: jika 1 motor masuk di pagi hari lalu keluar di sore hari, itu dihitung sebagai **2 kali pemeriksaan**. Metrik ini menggambarkan total intensitas buka-tutup palang gerbang.
        - 🎓 **Civitas Terdaftar**: Jumlah pemeriksaan terhadap kendaraan yang plat nomornya **sudah terdaftar di database master kampus** sebagai milik **Dosen** atau **Mahasiswa**, sehingga secara otomatis memperoleh hak tarif khusus (Dosen Rp0 / gratis, Mahasiswa tarif flat subsidi Rp2.000 seharian).
        - 🏷️ **Plat Tamu / Luar**: Jumlah pemeriksaan kendaraan tamu umum / luar kampus yang belum terdaftar di database kampus (dikenakan tarif normal tamu: Rp3.000 jam pertama + Rp2.000/jam berikutnya).
        - 🚗 **Sedang Parkir**: Jumlah kendaraan yang saat ini posisinya masih di dalam area kampus (sudah tercatat masuk tapi belum checkout keluar).
        """)

    st.divider()

    # --------------------------------------------------------------------------
    # 1. KENDARAAN SEDANG PARKIR DI DALAM KAMPUS (CRUD SESI AKTIF SATU PER SATU)
    # --------------------------------------------------------------------------
    st.markdown("### 🚨 Manajemen Kendaraan Sedang Parkir (CRUD Sesi Aktif)")
    st.caption("Kelola kendaraan (motor/mobil civitas maupun tamu) yang saat ini berada di dalam kampus satu per satu secara presisi (Lihat, Tambah Masuk, Edit, Checkout, atau Hapus).")

    crud_park1, crud_park2, crud_park3, crud_park4 = st.tabs([
        "📋 Daftar & Status Parkir (Read)",
        "➕ Catat Masuk Manual (Create)",
        "✏️ Edit Sesi Parkir (Update)",
        "🗑️ Checkout & Hapus Sesi (Delete Satu per Satu)",
    ])

    # CRUD 1: DAFTAR & STATUS PARKIR (READ & AKSI CEPAT SATU PER SATU)
    with crud_park1:
        active_parked_list = db_manager.get_currently_parked_vehicles()
        if active_parked_list:
            c_flt1, c_flt2 = st.columns([2, 1])
            with c_flt1:
                q_act = st.text_input("🔍 Cari Plat / Nama Pemilik di Dalam", "", key="search_active_parked_input")
            with c_flt2:
                kat_act = st.selectbox("Filter Kategori", ["Semua", "Dosen", "Mahasiswa", "Tamu / Belum Terdaftar"], key="filter_cat_active_select")

            filtered_active = []
            for a in active_parked_list:
                if q_act:
                    match_p = normalize_plate(q_act) in normalize_plate(a.get("plate", ""))
                    match_n = q_act.lower() in (a.get("owner_name") or "").lower()
                    if not (match_p or match_n):
                        continue
                if kat_act != "Semua":
                    if kat_act == "Tamu / Belum Terdaftar" and a.get("category") not in ["Tamu", "Belum terdaftar", None]:
                        continue
                    elif kat_act in ["Dosen", "Mahasiswa"] and a.get("category") != kat_act:
                        continue
                filtered_active.append(a)

            st.info(f"🟢 Terdeteksi **{len(active_parked_list)} unit kendaraan** yang saat ini masih berada di dalam area parkir kampus ({len(filtered_active)} ditampilkan):")
            
            active_table = []
            for a in filtered_active:
                active_table.append({
                    "ID": f"#{a.get('id')}",
                    "Plat Nomor": format_plate_display(a.get("plate")),
                    "Nama Pemilik": a.get("owner_name") or "(Tamu Luar)",
                    "Kategori": a.get("category") or "Tamu",
                    "Waktu Masuk": (a.get("created_at") or "").replace("T", " "),
                    "Durasi Berjalan": a.get("duration_text", "-"),
                    "Estimasi Tarif": rupiah(a.get("current_fee", 0)),
                    "Status": "🟢 Parkir di Dalam",
                })
            st.dataframe(pd.DataFrame(active_table), use_container_width=True, hide_index=True)

            # AKSI CEPAT SATU PER SATU
            st.markdown("##### ⚡ Aksi Langsung per Kendaraan (Satu per Satu):")
            st.caption("Pilih salah satu kendaraan di bawah untuk melakukan checkout atau menghapus sesi secara individual:")
            active_opts = {
                f"#{a['id']} | {format_plate_display(a['plate'])} ({a.get('owner_name') or a.get('category') or 'Tamu'})": a
                for a in active_parked_list
            }
            sel_active_label = st.selectbox(
                "Pilih Kendaraan Parkir Aktif:",
                list(active_opts.keys()),
                key="sel_active_veh_action"
            )
            sel_veh = active_opts[sel_active_label]
            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                if st.button(f"🚗 Checkout Keluar (OUT): {format_plate_display(sel_veh['plate'])}", type="primary", use_container_width=True, key="btn_checkout_single_active"):
                    db_manager.checkout_vehicle(sel_veh["plate"], notes="Checkout Manual Petugas via Dashboard")
                    st.toast(f"Kendaraan {format_plate_display(sel_veh['plate'])} berhasil dikeluarkan (OUT)!", icon="🚗")
                    st.rerun()
            with btn_c2:
                if st.button(f"🗑️ Hapus Sesi #{sel_veh['id']} ({format_plate_display(sel_veh['plate'])})", type="secondary", use_container_width=True, key="btn_del_single_active"):
                    db_manager.delete_active_parked_sessions(tx_id=sel_veh["id"])
                    st.toast(f"Sesi kendaraan #{sel_veh['id']} ({format_plate_display(sel_veh['plate'])}) berhasil dihapus satu per satu!", icon="🗑️")
                    st.rerun()
        else:
            st.success("✅ **Semua kendaraan telah keluar.** Saat ini tidak ada kendaraan yang sedang parkir di dalam area kampus.")

    # CRUD 2: CATAT MASUK MANUAL (CREATE)
    with crud_park2:
        st.markdown("#### ➕ Catat Kendaraan Masuk Manual (IN)")
        st.caption("Gunakan form ini jika ada kendaraan civitas atau tamu yang masuk tanpa terdeteksi kamera gerbang:")

        with st.form("form_create_active_park_session"):
            col_cr1, col_cr2 = st.columns(2)
            with col_cr1:
                cr_plat = st.text_input("Nomor Plat Kendaraan", placeholder="Contoh: B 1234 XYZ atau F 4526 FCF")
                cr_kat = st.selectbox("Kategori Pengendara", ["Mahasiswa", "Dosen", "Tamu / Pengunjung Luar"])
            with col_cr2:
                cr_nama = st.text_input("Nama Pengendara (Opsional)", placeholder="Nama civitas atau tamu luar")
                cr_notes = st.text_input("Catatan Petugas (Opsional)", value="Masuk Manual Petugas Pos")

            btn_submit_cr = st.form_submit_button("➕ Masukkan ke Parkir Aktif (IN)", type="primary", use_container_width=True)
            if btn_submit_cr:
                cleaned_plat = disambiguate_plate(cr_plat.strip())
                if not cleaned_plat:
                    st.error("Nomor plat kendaraan tidak boleh kosong!")
                elif not is_valid_plate_pattern(cleaned_plat):
                    st.error("Format nomor plat tidak valid. Masukkan format plat Indonesia yang benar (Contoh: B 1234 XYZ, F 4526 FCF).")
                else:
                    curr_active = db_manager.get_currently_parked_vehicles()
                    already_in = any(normalize_plate(v["plate"]) == normalize_plate(cleaned_plat) for v in curr_active)
                    if already_in:
                        st.warning(f"⚠️ Kendaraan dengan plat {format_plate_display(cleaned_plat)} sudah tercatat sedang berada di dalam area parkir kampus!")
                    else:
                        kat_val = "Tamu" if "Tamu" in cr_kat else cr_kat
                        db_manager.record_transaction(
                            plate=cleaned_plat,
                            owner_name=cr_nama.strip() or None,
                            category=kat_val,
                            fee=0,
                            gate_direction="IN",
                            source="Input Manual Petugas",
                            notes=cr_notes.strip() or None,
                        )
                        st.success(f"✅ Kendaraan {format_plate_display(cleaned_plat)} berhasil dicatat masuk (IN)!")
                        st.toast(f"Kendaraan {format_plate_display(cleaned_plat)} masuk!", icon="🚗")
                        st.rerun()

    # CRUD 3: EDIT SESI PARKIR (UPDATE)
    with crud_park3:
        st.markdown("#### ✏️ Edit Data Sesi Parkir Aktif")
        st.caption("Koreksi nomor plat (jika terjadi salah baca OCR kamera), ubah nama pemilik, atau perbarui kategori kendaraan yang sedang parkir:")

        active_parked_list = db_manager.get_currently_parked_vehicles()
        if active_parked_list:
            upd_choices = {
                f"#{a['id']} | {format_plate_display(a['plate'])} ({a.get('owner_name') or a.get('category') or 'Tamu'}) - Masuk: {(a.get('created_at') or '')[:16]}": a
                for a in active_parked_list
            }
            sel_upd_label = st.selectbox(
                "Pilih Kendaraan yang Ingin Diedit:",
                ["— Pilih Kendaraan —"] + list(upd_choices.keys()),
                key="sel_active_update_dropdown"
            )

            if sel_upd_label != "— Pilih Kendaraan —":
                target_veh = upd_choices[sel_upd_label]
                with st.form("form_update_active_session"):
                    col_u1, col_u2 = st.columns(2)
                    with col_u1:
                        u_plat = st.text_input("Nomor Plat", value=format_plate_display(target_veh["plate"]))
                        kat_opts = ["Mahasiswa", "Dosen", "Tamu"]
                        curr_k = target_veh.get("category") or "Tamu"
                        k_idx = kat_opts.index(curr_k) if curr_k in kat_opts else 2
                        u_kat = st.selectbox("Kategori", kat_opts, index=k_idx)
                    with col_u2:
                        u_nama = st.text_input("Nama Pemilik", value=target_veh.get("owner_name") or "")
                        u_notes = st.text_input("Catatan Petugas", value=target_veh.get("notes") or "")

                    btn_save_u = st.form_submit_button("💾 Simpan Perubahan Sesi", type="primary", use_container_width=True)
                    if btn_save_u:
                        cleaned_u = disambiguate_plate(u_plat.strip())
                        if not cleaned_u:
                            st.error("Nomor plat tidak boleh kosong!")
                        else:
                            ok = db_manager.update_transaction(
                                tx_id=target_veh["id"],
                                plate=cleaned_u,
                                owner_name=u_nama.strip() or None,
                                category=u_kat,
                                notes=u_notes.strip() or None,
                            )
                            if ok:
                                st.success(f"✅ Sesi kendaraan #{target_veh['id']} berhasil diperbarui!")
                                st.toast(f"Data {format_plate_display(cleaned_u)} diperbarui!", icon="💾")
                                st.rerun()
                            else:
                                st.error("Gagal memperbarui data sesi.")
        else:
            st.info("Saat ini tidak ada kendaraan yang sedang parkir untuk diedit.")

    # CRUD 4: CHECKOUT / HAPUS SESI (DELETE SATU PER SATU)
    with crud_park4:
        st.markdown("#### 🗑️ Checkout Keluar atau Hapus Sesi Parkir (Satu per Satu)")
        st.caption("Pilih kendaraan yang ingin diselesaikan atau dihapus dari daftar parkir aktif secara satu per satu. **Tindakan ini aman dan HANYA berdampak pada 1 kendaraan yang dipilih.**")

        active_parked_list = db_manager.get_currently_parked_vehicles()
        if active_parked_list:
            del_choices = {
                f"#{a['id']} | {format_plate_display(a['plate'])} ({a.get('owner_name') or a.get('category') or 'Tamu'}) - Masuk: {(a.get('created_at') or '')[:16]}": a
                for a in active_parked_list
            }
            sel_del_label = st.selectbox(
                "Pilih Kendaraan Parkir Aktif:",
                ["— Pilih Kendaraan —"] + list(del_choices.keys()),
                key="sel_active_del_tab_dropdown"
            )

            if sel_del_label != "— Pilih Kendaraan —":
                chosen_veh = del_choices[sel_del_label]

                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 1.5px solid rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 16px; margin: 12px 0;">
                    <h4 style="margin:0 0 8px 0; color: #60A5FA;">🚗 Detail Kendaraan Terpilih:</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; font-size: 0.93rem;">
                        <div><b>ID Transaksi:</b> #{chosen_veh['id']}</div>
                        <div><b>Nomor Plat:</b> <span style="font-family: monospace; font-size: 1.1rem; color: #FCD34D;">{format_plate_display(chosen_veh['plate'])}</span></div>
                        <div><b>Pemilik:</b> {chosen_veh.get('owner_name') or '(Tamu Luar)'}</div>
                        <div><b>Kategori:</b> {chosen_veh.get('category') or 'Tamu'}</div>
                        <div><b>Waktu Masuk:</b> {(chosen_veh.get('created_at') or '').replace('T', ' ')}</div>
                        <div><b>Durasi Berjalan:</b> {chosen_veh.get('duration_text', '-')}</div>
                        <div><b>Estimasi Tarif:</b> <span style="color: #34D399; font-weight: 700;">{rupiah(chosen_veh.get('current_fee', 0))}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c_act_out, c_act_del = st.columns(2)
                with c_act_out:
                    st.markdown("##### 🚪 Opsi A: Catat Checkout Keluar (Resmi)")
                    st.caption("Mencatat transaksi keluar (OUT) resmi, menghitung durasi total, dan mencatat tarif akhir ke laporan keuangan.")
                    if st.button(f"🚪 Checkout OUT #{chosen_veh['id']} ({format_plate_display(chosen_veh['plate'])})", type="primary", use_container_width=True, key="btn_confirm_checkout_out_one"):
                        db_manager.checkout_vehicle(chosen_veh["plate"], notes="Checkout Manual Petugas via CRUD Tab")
                        st.toast(f"Kendaraan {format_plate_display(chosen_veh['plate'])} berhasil dikeluarkan (OUT)!", icon="🚗")
                        st.rerun()
                with c_act_del:
                    st.markdown("##### 🗑️ Opsi B: Hapus Sesi Parkir (Satu per Satu)")
                    st.caption("Menghapus data sesi parkir ini secara permanen dari sistem (misal data testing kemarin atau salah input). Kendaraan lain tidak terpengaruh.")
                    if st.button(f"🗑️ Hapus Sesi #{chosen_veh['id']} Saja", type="secondary", use_container_width=True, key="btn_confirm_delete_active_one"):
                        db_manager.delete_active_parked_sessions(tx_id=chosen_veh["id"])
                        st.toast(f"Sesi #{chosen_veh['id']} ({format_plate_display(chosen_veh['plate'])}) berhasil dihapus satu per satu!", icon="🗑️")
                        st.rerun()
        else:
            st.success("✅ **Area parkir bersih.** Tidak ada kendaraan yang sedang parkir saat ini.")

    st.divider()

    # --------------------------------------------------------------------------
    # 2. CRUD LENGKAP LOG RIWAYAT TRANSAKSI
    # --------------------------------------------------------------------------
    st.markdown("### 📋 Manajemen & Riwayat Transaksi Gerbang")

    tx_list = db_manager.list_transactions(limit=1000)

    crud_hist1, crud_hist2, crud_hist3, crud_hist4 = st.tabs([
        "📋 Log Riwayat & Filter (Read)",
        "➕ Catat Transaksi Manual (Create)",
        "✏️ Edit Transaksi (Update)",
        "🗑️ Hapus & Pembersihan Log (Delete)",
    ])

    # --------------------------------------------------------------------------
    # CRUD 1: LOG RIWAYAT & FILTER (READ)
    # --------------------------------------------------------------------------
    with crud_hist1:
        r_col1, r_col2, r_col3, r_col4 = st.columns([2, 1, 1, 1])
        filter_plat_tx = r_col1.text_input("🔍 Cari Plat / Nama", "", key="filter_plat_tx_input")
        filter_cat_tx = r_col2.selectbox("Filter Kategori", ["Semua", "Dosen", "Mahasiswa", "Tamu / Belum Terdaftar"], key="filter_cat_tx_select")
        filter_gate_tx = r_col3.selectbox("Arah Gerbang", ["Semua", "🟢 Masuk (IN)", "🔴 Keluar (OUT)"], key="filter_gate_tx_select")
        limit_tx = r_col4.number_input("Batas Baris", min_value=10, max_value=1000, value=200, step=50, key="limit_tx_num")

        gate_filter_val = None
        if "IN" in filter_gate_tx:
            gate_filter_val = "IN"
        elif "OUT" in filter_gate_tx:
            gate_filter_val = "OUT"

        filtered_tx = db_manager.list_transactions(
            limit=limit_tx,
            plate_filter=filter_plat_tx,
            category_filter=filter_cat_tx if filter_cat_tx != "Semua" else None,
        )
        if gate_filter_val:
            filtered_tx = [t for t in filtered_tx if t.get("gate_direction") == gate_filter_val]

        if filtered_tx:
            for item in filtered_tx:
                raw_fee = item.get("fee")
                item["fee_display"] = rupiah(raw_fee)
                dur = item.get("duration_minutes")
                if item.get("gate_direction") == "OUT" and dur is not None and dur > 0:
                    item["duration_text"] = format_duration(dur)
                elif item.get("gate_direction") == "OUT":
                    item["duration_text"] = "< 1 Menit"
                else:
                    item["duration_text"] = "Sesi Masuk (IN)"

            df_tx = pd.DataFrame(filtered_tx)
            cols_order = ["id", "created_at", "plate", "owner_name", "category", "fee_display", "gate_direction", "duration_text", "decision", "notes"]
            display_tx = df_tx[cols_order]
            display_tx.columns = ["ID", "Waktu", "Plat Nomor", "Nama Pemilik", "Kategori", "Tarif", "Gerbang", "Durasi Parkir", "Status Keputusan", "Catatan"]
            
            st.dataframe(display_tx, use_container_width=True, hide_index=True)

            st.download_button(
                "📥 Unduh Laporan Riwayat (CSV)",
                data=display_tx.to_csv(index=False).encode("utf-8-sig"),
                file_name="laporan_transaksi_parkir.csv",
                mime="text/csv",
                key="btn_download_tx_csv"
            )
        else:
            st.info("Belum ada catatan transaksi yang sesuai filter.")

    # --------------------------------------------------------------------------
    # CRUD 2: CATAT TRANSAKSI MANUAL (CREATE)
    # --------------------------------------------------------------------------
    with crud_hist2:
        st.markdown("#### ➕ Catat Transaksi Masuk / Keluar Manual")
        st.caption("Gunakan jika petugas membuka palang gerbang secara manual tanpa melalui kamera otomatis.")

        with st.form("form_create_manual_tx", clear_on_submit=True):
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                man_plat = st.text_input("Nomor Plat Kendaraan", placeholder="Contoh: B 1234 XYZ")
                man_cat = st.selectbox("Kategori Civitas", ["Mahasiswa", "Dosen", "Tamu / Belum Terdaftar"])
                man_gate = st.selectbox("Arah Gerbang", ["IN (Masuk)", "OUT (Keluar)"])
            with col_m2:
                man_owner = st.text_input("Nama Pemilik (Opsional)", placeholder="Nama Pengendara")
                man_fee = st.number_input("Tarif Parkir (Rp)", min_value=0, max_value=500000, value=2000, step=1000)
                man_notes = st.text_input("Catatan Petugas (Opsional)", placeholder="Contoh: Palang dibuka manual oleh petugas pos")

            btn_submit_man = st.form_submit_button("➕ Simpan Transaksi Manual", type="primary", use_container_width=True)
            if btn_submit_man:
                if not man_plat.strip():
                    st.error("Nomor plat kendaraan wajib diisi!")
                else:
                    gate_code = "IN" if "IN" in man_gate else "OUT"
                    saved_id = db_manager.save_transaction(
                        plate=man_plat.strip(),
                        owner_name=man_owner.strip() or None,
                        category=man_cat,
                        fee=int(man_fee),
                        decision=f"INPUT MANUAL PETUGAS ({gate_code})",
                        status="CONFIRMED",
                        gate_direction=gate_code,
                        notes=man_notes.strip() or "Input Manual",
                    )
                    st.success(f"🎉 Transaksi #{saved_id} ({format_plate_display(man_plat)}) berhasil dicatat secara manual!")
                    st.rerun()

    # --------------------------------------------------------------------------
    # CRUD 3: EDIT TRANSAKSI (UPDATE)
    # --------------------------------------------------------------------------
    with crud_hist3:
        st.markdown("#### ✏️ Edit / Perbarui Data Transaksi")
        st.caption("Koreksi data riwayat transaksi jika terjadi salah input plat, perubahan tarif, atau koreksi catatan.")

        if tx_list:
            tx_edit_choices = {
                f"#{t['id']} | {format_plate_display(t['plate'])} ({t.get('gate_direction')}) - {t.get('category')} - {t['created_at'][:16]}": t
                for t in tx_list[:100]
            }
            sel_edit_tx_label = st.selectbox(
                "Pilih Transaksi yang Ingin Diedit:",
                ["— Pilih Transaksi —"] + list(tx_edit_choices.keys()),
                key="select_edit_tx_dropdown"
            )

            if sel_edit_tx_label != "— Pilih Transaksi —":
                target_tx = tx_edit_choices[sel_edit_tx_label]

                with st.form("form_edit_existing_tx"):
                    col_ed1, col_ed2 = st.columns(2)
                    with col_ed1:
                        ed_plat = st.text_input("Nomor Plat", value=format_plate_display(target_tx["plate"]))
                        cat_list = ["Mahasiswa", "Dosen", "Tamu / Belum Terdaftar"]
                        cat_idx = 0
                        if target_tx.get("category") in cat_list:
                            cat_idx = cat_list.index(target_tx["category"])
                        ed_cat = st.selectbox("Kategori", cat_list, index=cat_idx)
                        ed_gate_idx = 0 if target_tx.get("gate_direction") == "IN" else 1
                        ed_gate = st.selectbox("Arah Gerbang", ["IN (Masuk)", "OUT (Keluar)"], index=ed_gate_idx)
                    with col_ed2:
                        ed_owner = st.text_input("Nama Pemilik", value=target_tx.get("owner_name") or "")
                        cur_fee = target_tx.get("fee") if target_tx.get("fee") is not None else 0
                        ed_fee = st.number_input("Tarif Parkir (Rp)", min_value=0, max_value=500000, value=int(cur_fee), step=1000)
                        ed_notes = st.text_input("Catatan Petugas", value=target_tx.get("notes") or "")

                    btn_save_tx_edit = st.form_submit_button("💾 Simpan Perubahan Transaksi", type="primary", use_container_width=True)
                    if btn_save_tx_edit:
                        if not ed_plat.strip():
                            st.error("Nomor plat tidak boleh kosong!")
                        else:
                            gate_dir = "IN" if "IN" in ed_gate else "OUT"
                            ok = db_manager.update_transaction(
                                tx_id=target_tx["id"],
                                plate=ed_plat.strip(),
                                owner_name=ed_owner.strip() or None,
                                category=ed_cat,
                                fee=int(ed_fee),
                                gate_direction=gate_dir,
                                notes=ed_notes.strip() or None,
                            )
                            if ok:
                                st.success(f"✅ Transaksi #{target_tx['id']} berhasil diperbarui!")
                                st.rerun()
                            else:
                                st.error("Gagal memperbarui transaksi.")
        else:
            st.info("Belum ada transaksi di database untuk diedit.")

    # --------------------------------------------------------------------------
    # CRUD 4: HAPUS & PEMBERSIHAN LOG (DELETE)
    # --------------------------------------------------------------------------
    with crud_hist4:
        st.markdown("#### 🗑️ Hapus & Bersihkan Riwayat Transaksi")
        st.caption("Pilih opsi penghapusan riwayat transaksi di bawah ini agar data tidak menumpuk:")

        m_col1, m_col2 = st.columns(2)

        # Opsi 1: Hapus Transaksi Tertentu Berdasarkan ID
        with m_col1:
            st.markdown("##### 🗑️ 1. Hapus Transaksi Tertentu (Per ID)")
            st.caption("Hapus satu data transaksi tertentu jika terjadi salah scan kamera atau transaksi palsu.")
            if tx_list:
                tx_choices_del = {
                    f"#{t['id']} | {format_plate_display(t['plate'])} ({t.get('gate_direction')}) - {t['created_at'][:16]}": t["id"]
                    for t in tx_list[:50]
                }
                del_choice_label = st.selectbox(
                    "Pilih Transaksi yang Ingin Dihapus:",
                    ["— Pilih Transaksi —"] + list(tx_choices_del.keys()),
                    key="select_del_tx_id_tab4"
                )
                if del_choice_label != "— Pilih Transaksi —":
                    target_tx_id = tx_choices_del[del_choice_label]
                    if st.button(f"🗑️ Hapus Transaksi #{target_tx_id} Sekarang", type="secondary", use_container_width=True, key="btn_del_single_tx_tab4"):
                        db_manager.delete_transaction(target_tx_id)
                        st.toast(f"Transaksi #{target_tx_id} berhasil dihapus!", icon="🗑️")
                        st.rerun()
            else:
                st.caption("Tidak ada riwayat untuk dihapus.")

        # Opsi 2 & 3: Bersihkan Riwayat Selesai & Reset Total
        with m_col2:
            st.markdown("##### 🧹 2. Bersihkan Riwayat Selesai (Hanya OUT)")
            st.caption("Menghapus log transaksi yang kendaraannya sudah keluar (OUT). **Sesi kendaraan yang sedang parkir (IN) tetap aman**.")
            if st.button("🧹 Bersihkan Transaksi Selesai", type="primary", use_container_width=True, key="btn_clean_completed_tx_tab4"):
                deleted_count = db_manager.delete_completed_transactions()
                st.toast(f"Berhasil membersihkan {deleted_count} riwayat transaksi selesai!", icon="🧹")
                st.rerun()

            st.divider()
            st.markdown("##### ⚠️ 3. Reset Seluruh Riwayat Database")
            st.caption("Menghapus SELURUH catatan riwayat transaksi di database (reset total riwayat).")
            confirm_reset_all = st.checkbox("Saya yakin ingin menghapus seluruh log transaksi secara permanen", key="chk_confirm_reset_all_tx_tab4")
            if st.button("⚠️ Hapus Seluruh Riwayat", type="secondary", disabled=not confirm_reset_all, use_container_width=True, key="btn_clear_all_tx_danger_tab4"):
                cleared_count = db_manager.clear_all_transactions()
                st.toast(f"Seluruh {cleared_count} transaksi riwayat berhasil direset!", icon="⚠️")
                st.rerun()


# ==============================================================================
# TAB 4: PANDUAN SISTEM & STATUS AI
# ==============================================================================
with tab_guide:
    ai_status = {
        "yolo": os.path.exists(DEFAULT_MODEL_PATH),
        "ocr": True,
        "database": os.path.exists(os.path.join(APP_DIR, "parking.db")),
        "camera": True,
    }
    render_panduan_ai(status=ai_status)

    st.divider()
    st.markdown("### 📊 Status Dataset & Model YOLO Saat Ini")
    
    rep = validate_dataset(DATASET_DIR)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gambar Training", rep["train_images_count"])
    c2.metric("Label Training", rep["train_labels_count"])
    c3.metric("Gambar Validasi", rep["val_images_count"])
    c4.metric("Label Validasi", rep["val_labels_count"])

    if rep["is_valid"]:
        st.success(f"✅ Dataset valid! Model aktif di: `{DEFAULT_MODEL_PATH}`")
    else:
        st.info("Dataset lokal siap untuk penambahan data baru.")

    st.markdown("#### 🏋️ Cara Melatih Ulang Model (Jika Ada Foto Baru):")
    st.markdown("""
Model YOLO membutuhkan **gambar** dan **file label koordinat kotak plat (`.txt`)**. Jika Anda dan tim memiliki foto-foto motor baru:

1. **Beri Label di Roboflow:** Unggah foto baru ke [Roboflow.com](https://roboflow.com/), tarik kotak pada plat nomor (label: `license_plate`).
2. **Export Dataset:** Pilih format **YOLOv8 / YOLOv11 PyTorch**, lalu unduh file ZIP.
3. **Ekstrak ke Folder Proyek:** Ekstrak file gambar ke `dataset/images/` dan file label teks ke `dataset/labels/`.
4. **Jalankan Training:** Buka terminal dan jalankan:
    """)
    st.code("python train_model.py --epochs 60 --batch 8 --auto-copy", language="powershell")
    st.caption("💡 *Tips Praktis:* Jika tidak ingin repot melabeli di Roboflow, Anda juga bisa menaruh foto di folder `DATA SET APLIKASI` dan meminta asisten AI untuk memproses label koordinatnya secara otomatis!")
