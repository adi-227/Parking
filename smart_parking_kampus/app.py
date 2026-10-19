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
# TEMA & GAYA VISUAL (ADAPTIVE LIGHT & DARK THEME — PREMIUM CAMPUS DASHBOARD)
# ==============================================================================
if "app_theme" not in st.session_state:
    st.session_state["app_theme"] = "Dark"

is_dark = (st.session_state.get("app_theme", "Dark") == "Dark")

if is_dark:
    theme_vars = """
        --bg-canvas: #0E1117;
        --bg-card: #161B26;
        --bg-sidebar: #11151F;
        --border-color: rgba(255, 255, 255, 0.12);
        --text-primary: #F3F4F6;
        --text-secondary: #94A3B8;
        --tab-list-bg: #161B26;
        --tab-btn-bg: #0E1117;
        --tab-btn-text: #F3F4F6;
        --tab-btn-border: rgba(255, 255, 255, 0.14);
        --input-bg: #161B26;
        --input-text: #F3F4F6;
        --input-border: rgba(255, 255, 255, 0.16);
    """
else:
    theme_vars = """
        --bg-canvas: #F8FAFC;
        --bg-card: #FFFFFF;
        --bg-sidebar: #F1F5F9;
        --border-color: rgba(0, 0, 0, 0.12);
        --text-primary: #0F172A;
        --text-secondary: #64748B;
        --tab-list-bg: #F1F5F9;
        --tab-btn-bg: #FFFFFF;
        --tab-btn-text: #1E293B;
        --tab-btn-border: rgba(0, 0, 0, 0.15);
        --input-bg: #FFFFFF;
        --input-text: #0F172A;
        --input-border: rgba(0, 0, 0, 0.16);
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

    /* Page Canvas: Menyesuaikan Otomatis Mode Light & Dark */
    .stApp {{
        background-color: var(--bg-canvas) !important;
        color: var(--text-primary) !important;
    }}

    /* Header Bar */
    header[data-testid="stHeader"] {{
        background-color: var(--bg-canvas) !important;
    }}
    header[data-testid="stHeader"] * {{
        color: var(--text-primary) !important;
    }}

    /* Input & Select Controls */
    div[data-baseweb="input"] > div,
    input, textarea {{
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        border-color: var(--input-border) !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        border-color: var(--input-border) !important;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .main .block-container {
        max-width: 1320px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
    }

    /* ==========================================================================
       CARD NAVIGATION BAR (TAB LIST UTAMA JADI CARD BESAR & MEWAH)
       ========================================================================== */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.22) !important;
        border-radius: 16px !important;
        padding: 8px !important;
        gap: 12px !important;
        margin-top: 8px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08) !important;
        display: flex !important;
        width: 100% !important;
        flex-wrap: wrap !important;
    }

    /* HAPUS SEMUA UNDERLINE / BORDER BAWAAN STREAMLIT */
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-border"],
    [data-baseweb="tab-highlight"],
    [data-baseweb="tab-border"],
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"] {
        display: none !important;
        height: 0px !important;
        opacity: 0 !important;
        visibility: hidden !important;
        border: none !important;
    }

    /* Setiap Tombol Tab di dalam Card jadi Kartu Navigasi yang Besar & Empuk */
    [data-testid="stTabs"] button[data-baseweb="tab"],
    [data-testid="stTabs"] [data-baseweb="tab-list"] button {
        flex: 1 1 200px !important;
        background-color: var(--background-color) !important;
        border: 1.5px solid rgba(128, 128, 128, 0.22) !important;
        border-radius: 12px !important;
        padding: 14px 20px !important;
        color: var(--text-color) !important;
        font-size: 0.98rem !important;
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
    }

    /* Hover Tab Card */
    [data-testid="stTabs"] button[data-baseweb="tab"]:hover,
    [data-testid="stTabs"] [data-baseweb="tab-list"] button:hover {
        border-color: #2563EB !important;
        color: #2563EB !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.2) !important;
    }

    /* Active Tab Card: Gradien Biru Mewah dengan Glow */
    [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
    [data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
        border: 1.5px solid #3B82F6 !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Teks Tab Aktif WAJIB Putih */
    [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] * {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #091e42 0%, #172554 45%, #1d4ed8 100%);
        padding: 22px 28px;
        border-radius: 16px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 10px 25px -5px rgba(29, 78, 216, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff !important;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #bfdbfe !important;
        margin-top: 4px;
        margin-bottom: 0px;
        font-weight: 400;
    }
    .hero-chips {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 14px;
    }
    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.14);
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #eff6ff;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Stepper Workflow Visual */
    .stepper-box {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 14px;
        padding: 12px 18px;
        margin-bottom: 20px;
        gap: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        color: var(--text-color);
    }
    .step-card {
        display: flex;
        align-items: center;
        gap: 10px;
        flex: 1;
        padding: 8px 12px;
        border-radius: 10px;
        transition: all 0.2s ease;
    }
    .step-card.active {
        background: rgba(37, 99, 235, 0.12) !important;
        border: 1.5px solid #2563eb !important;
        box-shadow: 0 0 12px rgba(37, 99, 235, 0.2);
    }
    .step-card.completed {
        background: rgba(16, 185, 129, 0.12) !important;
        border: 1.5px solid #10b981 !important;
    }
    .step-card.idle {
        opacity: 0.55;
        border: 1px solid transparent;
    }
    .step-num {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.92rem;
        flex-shrink: 0;
    }
    .step-card.active .step-num {
        background: #2563eb;
        color: #ffffff;
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.5);
    }
    .step-card.completed .step-num {
        background: #10b981;
        color: #ffffff;
    }
    .step-card.idle .step-num {
        background: rgba(128, 128, 128, 0.2);
        color: inherit;
    }
    .step-meta {
        display: flex;
        flex-direction: column;
    }
    .step-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        opacity: 0.75;
    }
    .step-title {
        font-size: 0.88rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .step-arrow {
        opacity: 0.6;
        font-weight: bold;
        font-size: 1.1rem;
        user-select: none;
    }

    /* Radio Selector Metode Pindai jadi Kartu-Kartu Modern */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: grid !important;
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 10px !important;
        margin-top: 10px !important;
        margin-bottom: 16px !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.22) !important;
        border-radius: 12px !important;
        padding: 12px 14px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        color: var(--text-color) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        border-color: rgba(37, 99, 235, 0.5) !important;
        background: rgba(37, 99, 235, 0.08) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        border: 1.5px solid #2563EB !important;
        background: rgba(37, 99, 235, 0.12) !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.2) !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color) !important;
        padding: 14px 18px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.15) !important;
        border-color: rgba(37, 99, 235, 0.4) !important;
    }

    /* Ticket Card Luxury Modern */
    .ticket-card {
        background: var(--secondary-background-color) !important;
        border: 2px solid #2563eb !important;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.18);
        position: relative;
        overflow: hidden;
        color: var(--text-color);
    }
    .ticket-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #2563eb, #38bdf8, #10b981);
    }
    .ticket-card-invalid {
        background: var(--secondary-background-color) !important;
        border: 2px solid #ef4444 !important;
        border-radius: 16px;
        padding: 26px 20px;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.18);
        color: var(--text-color);
    }
    .empty-state-card {
        background: var(--secondary-background-color) !important;
        border: 2px dashed rgba(128, 128, 128, 0.3) !important;
        border-radius: 16px;
        padding: 36px 20px;
        text-align: center;
        margin-bottom: 16px;
        color: var(--text-color);
    }
    .empty-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 4px;
        color: var(--text-color);
    }
    .empty-sub {
        font-size: 0.88rem;
        opacity: 0.75;
        max-width: 440px;
        margin: 0 auto 16px auto;
        line-height: 1.5;
        color: var(--text-color);
    }
    .empty-pills {
        display: flex;
        gap: 8px;
        justify-content: center;
        flex-wrap: wrap;
    }
    .pill-mini {
        background: rgba(128, 128, 128, 0.15);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-color);
    }

    /* Authentic Indonesian License Plate */
    .plate-badge-id {
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
    }

    /* Badges Status */
    .badge-status {
        display: inline-block;
        padding: 7px 16px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: 0.2px;
    }
    .status-dosen { background: rgba(16, 185, 129, 0.18); color: #10B981; border: 1.5px solid rgba(16, 185, 129, 0.4); }
    .status-mhs { background: rgba(37, 99, 235, 0.18); color: #2563EB; border: 1.5px solid rgba(37, 99, 235, 0.4); }
    .status-tamu { background: rgba(245, 158, 11, 0.18); color: #D97706; border: 1.5px solid rgba(245, 158, 11, 0.4); }
    .status-invalid { background: rgba(239, 68, 68, 0.18); color: #DC2626; border: 1.5px solid rgba(239, 68, 68, 0.4); }
    .status-uncertain { background: rgba(234, 179, 8, 0.18); color: #CA8A04; border: 1.5px solid rgba(234, 179, 8, 0.4); }

    /* Action Buttons */
    div.stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 12px 20px !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
        border: 1px solid #3B82F6 !important;
        color: white !important;
        font-weight: 800 !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35) !important;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.5) !important;
        border-color: #60A5FA !important;
    }

    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="baseButton-secondary"] {
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        color: var(--text-color) !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        border-color: #2563EB !important;
        transform: translateY(-1px) !important;
    }

    /* Forms and Tables */
    div[data-testid="stForm"] {
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 16px !important;
        padding: 22px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05) !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px !important;
        overflow: hidden !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            flex-direction: column !important;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            grid-template-columns: 1fr !important;
        }
        .hero-title {
            font-size: 1.35rem;
        }
    }
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
    st.markdown("### 🌓 Mode Tampilan Web")
    curr_theme = st.session_state.get("app_theme", "Dark")
    theme_choice = st.radio(
        "Pilih Tema:",
        ["🌙 Gelap (Dark)", "☀️ Terang (Light)"],
        index=0 if curr_theme == "Dark" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="sidebar_theme_toggle_radio",
    )
    new_theme = "Light" if "Terang" in theme_choice else "Dark"
    if new_theme != curr_theme:
        st.session_state["app_theme"] = new_theme
        st.rerun()

    st.divider()
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
    - **👨‍🏫 Dosen & Tendik**: Gratis (Rp 0)
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
                "📁 Unggah Foto Kendaraan",
                "⌨️ Ketik Plat Manual",
                "📹 Kamera Live (Streaming WebRTC)",
            ],
            horizontal=False,
            label_visibility="collapsed",
        )

        # METODE 1: KAMERA SNAPSHOT (NATIVE BROWSER HTML5 - 60 FPS, 0% LAG)
        if mode_input == "⚡ Kamera Cepat (Snapshot 1-Klik)":
            if not is_model_available:
                st.warning("⚠️ File model YOLO belum ditemukan. Gunakan mode 'Unggah Foto' atau 'Ketik Manual'.")
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

        # METODE 2: UNGGAH FOTO
        elif mode_input == "📁 Unggah Foto Kendaraan":
            st.info("💡 Unggah foto bagian depan / belakang mobil atau motor:")
            uploaded_file = st.file_uploader("Pilih file foto (JPG/PNG)", type=["jpg", "jpeg", "png"], key="gate_photo_upload")
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Foto Kendaraan", use_container_width=True)

                if st.button("🔍 Deteksi & Baca Plat Nomor", type="primary", use_container_width=True):
                    with st.spinner("AI menganalisis gambar kendaraan..."):
                        from ocr_engine import extract_plate_text
                        found = []
                        ih, iw = img_cv.shape[:2]
                        if is_model_available:
                            model, reader = get_yolo_and_ocr(DEFAULT_MODEL_PATH)
                            eff_conf = min(conf_threshold, 0.25)
                            preds = model.predict(source=img_cv, conf=eff_conf, verbose=False, imgsz=640)
                            candidates = extract_candidate_plate_boxes(preds[0].boxes if preds else [], frame_w=iw, frame_h=ih)
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
                            st.session_state["current_source"] = "Unggah Foto"
                            st.toast(f"Plat terdeteksi: {best_t}", icon="✅")
                            st.rerun()
                        else:
                            st.session_state["current_plate"] = "INVALID_PLATE"
                            st.session_state["current_conf"] = 0.0
                            st.session_state["current_source"] = "Bukan Kendaraan"
                            st.rerun()

        # METODE 3: KETIK MANUAL
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
                    Gunakan kamera snapshot 1-klik, unggah foto, atau ketik plat di sebelah kiri.<br>
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
                    f"<div>🕒 <b>Waktu Masuk:</b><br><span style='font-size:0.92rem; color:#475569;'>{entry_disp}</span></div>"
                )
            else:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                extra_gate_info = (
                    "<div>⏱️ <b>Status Sesi:</b><br><span style='font-size:1.1rem; font-weight:700; color:#2563eb;'>Tiket Baru Masuk</span></div>"
                    f"<div>🕒 <b>Waktu Masuk:</b><br><span style='font-size:0.92rem; color:#475569;'>{now_str}</span></div>"
                )

            fee_note = ""
            if "Tamu" in str(dec_result.get("category", "")):
                if gate_code == "IN":
                    fee_note = "<div style='font-size:0.75rem; color:#64748b;'>Tarif 1 Jam Pertama (+Rp2.000/jam)</div>"
                elif duration_minutes > 60:
                    extra_h = (duration_minutes - 1) // 60
                    fee_note = f"<div style='font-size:0.75rem; color:#64748b;'>1 Jam Pertama (Rp3.000) + {extra_h} Jam x Rp2.000</div>"
                else:
                    fee_note = "<div style='font-size:0.75rem; color:#64748b;'>Tarif 1 Jam Pertama (<= 60 Menit)</div>"
            elif "Mahasiswa" in str(dec_result.get("category", "")):
                fee_note = "<div style='font-size:0.75rem; color:#64748b;'>Tarif Flat Subsidi Kampus (Seharian)</div>"
            elif "Dosen" in str(dec_result.get("category", "")):
                fee_note = "<div style='font-size:0.75rem; color:#10b981;'>Akses Sivitas Akademika Gratis</div>"

            # Kartu Tiket Digital Modern
            card_html = f"""
            <div class="ticket-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px;">
                    <div>
                        <div style="font-size:0.75rem; color:#64748b; font-weight:700; letter-spacing:0.5px; text-transform:uppercase;">Plat Nomor Terbaca:</div>
                        <div class="plate-badge-id">{format_plate_display(cleaned_plat)}</div>
                        <div style="font-size:0.82rem; color:#64748b;">Sumber: <b>{st.session_state.get('current_source')}</b> (OCR Conf: {st.session_state.get('current_conf', 1.0):.0%})</div>
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
# TAB 2: DATA KENDARAAN KAMPUS (ADMIN DATABASE)
# ==============================================================================
with tab_vehicles:
    st.subheader("Data Master Kendaraan Terdaftar (Civitas Akademika)")
    st.caption("Pihak kampus mendaftarkan Dosen dan Mahasiswa di sini agar mendapatkan hak parkir resmi / gratis.")

    col_add, col_csv = st.columns([1, 1])

    with col_add:
        st.markdown("#### ➕ Tambah / Perbarui Kendaraan")
        with st.form("form_tambah_kendaraan", clear_on_submit=True):
            in_plat = st.text_input("Nomor Plat", placeholder="Contoh: B 1234 XYZ")
            in_nama = st.text_input("Nama Pemilik", placeholder="Nama Dosen atau Mahasiswa")
            in_kat = st.selectbox("Kategori", ["Dosen", "Mahasiswa"])
            submit_btn = st.form_submit_button("Simpan ke Database", type="primary")

            if submit_btn:
                try:
                    is_up, msg = db_manager.upsert_vehicle(in_plat, in_nama, in_kat)
                    st.success(msg)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Gagal menyimpan: {exc}")

    with col_csv:
        st.markdown("#### 📥 Impor & Ekspor Data CSV")
        st.caption("Unggah file CSV dengan kolom: `plate, owner_name, category`")
        uploaded_csv = st.file_uploader("Pilih file CSV", type=["csv"], key="uploader_csv_v")
        
        if uploaded_csv is not None:
            try:
                df_csv = pd.read_csv(uploaded_csv, dtype=str).fillna("")
                req_cols = {"plate", "owner_name", "category"}
                if not req_cols.issubset(set(df_csv.columns)):
                    st.error("Kolom wajib CSV: plate, owner_name, category.")
                else:
                    st.dataframe(df_csv.head(3), use_container_width=True, hide_index=True)
                    if st.button("Proses Impor CSV"):
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

    st.divider()
    st.markdown("#### 📋 Daftar Kendaraan Terdaftar")

    f_col1, f_col2 = st.columns([2, 1])
    search_q = f_col1.text_input("🔍 Cari Plat / Nama Pemilik", "")
    filter_k = f_col2.selectbox("Filter Kategori", ["Semua", "Dosen", "Mahasiswa"])

    kendaraan_list = db_manager.list_vehicles(query=search_q, category_filter=filter_k)

    if kendaraan_list:
        df_display = pd.DataFrame(kendaraan_list)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        col_act1, col_act2 = st.columns([1, 1])
        with col_act1:
            csv_exp = df_display.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Ekspor Daftar Kendaraan ke CSV", data=csv_exp, file_name="data_kendaraan_kampus.csv", mime="text/csv")
        
        with col_act2:
            plat_to_delete = st.selectbox("Pilih Plat yang Ingin Dihapus:", ["— Pilih —"] + [k["plate"] for k in kendaraan_list])
            if plat_to_delete != "— Pilih —":
                if st.button(f"🗑️ Hapus {plat_to_delete}", type="secondary"):
                    db_manager.delete_vehicle(plat_to_delete)
                    st.success(f"Kendaraan {plat_to_delete} berhasil dihapus.")
                    st.rerun()
    else:
        st.info("Belum ada kendaraan yang cocok dengan pencarian.")

# ==============================================================================
# TAB 3: LAPORAN & RIWAYAT TRANSAKSI
# ==============================================================================
with tab_history:
    st.subheader("Dashboard & Log Riwayat Transaksi")

    metrics = db_manager.get_dashboard_metrics()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Kendaraan Terdaftar", metrics["total_vehicles_registered"])
    m2.metric("Sedang Parkir", f"{metrics['currently_parked']} Unit")
    m3.metric("Total Pemeriksaan", metrics["total_examinations"])
    m4.metric("Civitas Terdaftar", metrics["total_registered_inspected"])
    m5.metric("Plat Tamu / Luar", metrics["total_unregistered_inspected"])
    m6.metric("Total Pendapatan", rupiah(metrics["total_revenue"]))

    st.divider()
    r_col1, r_col2, r_col3 = st.columns([2, 1, 1])
    filter_plat_tx = r_col1.text_input("Filter Plat Transaksi", "")
    filter_cat_tx = r_col2.selectbox("Filter Kategori", ["Semua", "Dosen", "Mahasiswa", "Tamu / Belum Terdaftar"])
    limit_tx = r_col3.number_input("Batas Baris", min_value=10, max_value=1000, value=200, step=50)

    tx_list = db_manager.list_transactions(
        limit=limit_tx,
        plate_filter=filter_plat_tx,
        category_filter=filter_cat_tx if filter_cat_tx != "Semua" else None,
    )

    if tx_list:
        for item in tx_list:
            raw_fee = item.get("fee")
            item["fee_display"] = rupiah(raw_fee)
            dur = item.get("duration_minutes")
            if item.get("gate_direction") == "OUT" and dur is not None and dur > 0:
                item["duration_text"] = format_duration(dur)
            elif item.get("gate_direction") == "OUT":
                item["duration_text"] = "< 1 Menit"
            else:
                item["duration_text"] = "Sesi Masuk"

        df_tx = pd.DataFrame(tx_list)
        cols_order = ["id", "created_at", "plate", "owner_name", "category", "fee_display", "gate_direction", "duration_text", "decision", "notes"]
        display_tx = df_tx[cols_order]
        display_tx.columns = ["ID", "Waktu", "Plat Nomor", "Nama Pemilik", "Kategori", "Tarif", "Gerbang", "Durasi Parkir", "Status Keputusan", "Catatan"]
        
        st.dataframe(display_tx, use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Unduh Laporan Riwayat (CSV)",
            data=display_tx.to_csv(index=False).encode("utf-8-sig"),
            file_name="laporan_transaksi_parkir.csv",
            mime="text/csv",
        )
    else:
        st.info("Belum ada catatan transaksi yang sesuai filter.")

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
