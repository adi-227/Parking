import os
import re
import cv2
import threading
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

from parking_logic import (
    normalize_plate,
    format_plate_display,
    disambiguate_plate,
    is_valid_plate_pattern,
    format_duration,
    decide,
    rupiah,
    TARIFFS,
    CATEGORIES,
)
import db_manager
from dataset_validator import validate_dataset

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
# TEMA & GAYA VISUAL (MODERN CAMPUS DASHBOARD)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hero Header Compact */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
        padding: 20px 26px;
        border-radius: 14px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px -4px rgba(37, 99, 235, 0.25);
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff !important;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 0.9rem;
        color: #bfdbfe !important;
        margin-top: 4px;
        margin-bottom: 0px;
        font-weight: 400;
    }
    .badge-pipeline {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 8px;
        color: #e0f2fe;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.15);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
    }

    /* Verification Result Card */
    .ticket-card {
        background-color: var(--secondary-background-color);
        border: 2px solid rgba(37, 99, 235, 0.25);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
    }
    .plate-number-badge {
        display: inline-block;
        background: #0f172a;
        color: #ffffff;
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.8rem;
        font-weight: 800;
        padding: 6px 18px;
        border-radius: 8px;
        border: 2px solid #ffffff;
        letter-spacing: 2px;
        margin-bottom: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }

    /* Badges Status */
    .badge-status {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
    }
    .status-dosen { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .status-mhs { background: #e0f2fe; color: #075985; border: 1px solid #7dd3fc; }
    .status-tamu { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .status-invalid { background: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
    .status-uncertain { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }

    /* Clean Buttons */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>

<div class="hero-container">
    <div class="hero-title">🅿️ SMART PARKING KAMPUS</div>
    <div class="hero-subtitle">Sistem Otomasi Gerbang Parkir: Deteksi Plat Nomor (YOLO) + Pembacaan Karakter (EasyOCR) + Verifikasi Database Kampus</div>
    <div class="badge-pipeline">⚡ Alur Resmi: Deteksi Plat ➔ OCR Karakter ➔ Pencocokan SQLite ➔ Konfirmasi Petugas</div>
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

    is_model_available = os.path.isfile(DEFAULT_MODEL_PATH)
    if is_model_available:
        st.success("✅ Model YOLO Aktif")
    else:
        st.warning("⚠️ Model Belum Terpasang")

    st.divider()
    st.subheader("🅿️ Kapasitas Slot Parkir")
    total_slots = 100
    parked_now = db_manager.get_currently_parked_count()
    available_slots = max(0, total_slots - parked_now)
    pct_used = min(1.0, parked_now / total_slots)

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
    for cat, fee in TARIFFS.items():
        st.write(f"- **{cat}**: {rupiah(fee)}")

    st.divider()
    with st.expander("⚙️ Parameter Teknis & AI (Lanjutan)", expanded=False):
        st.caption("Pengaturan teknis ini sudah dioptimalkan secara default:")
        model_path_input = st.text_input("Path Model YOLO", value=DEFAULT_MODEL_PATH)
        is_model_available = os.path.isfile(model_path_input)
        
        conf_threshold = st.slider("Confidence YOLO", 0.10, 0.90, 0.25, 0.05)
        ocr_min_conf = st.slider("Confidence Min OCR", 0.20, 0.80, 0.40, 0.05)
        frame_skip = st.slider("Frekuensi Frame (Webcam)", 2, 16, 6, 2)
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
    st.subheader("Pemeriksaan Kendaraan di Pintu Masuk / Keluar")
    st.caption("Pilih metode deteksi plat di bawah ini. Sistem akan otomatis memverifikasi hak parkir ke database kampus.")

    # State untuk menyimpan plat terakhir yang dideteksi
    if "current_plate" not in st.session_state:
        st.session_state["current_plate"] = None
    if "current_conf" not in st.session_state:
        st.session_state["current_conf"] = 1.0
    if "current_source" not in st.session_state:
        st.session_state["current_source"] = "Belum Ada"

    col_input, col_verify = st.columns([1.1, 1.2])

    with col_input:
        mode_input = st.radio(
            "Pilih Metode Deteksi:",
            ["📹 Kamera Langsung (Webcam)", "📁 Unggah Foto Kendaraan", "⌨️ Ketik Plat Manual"],
            horizontal=True,
        )

        # MODE 1: KAMERA WEBCAM
        if mode_input == "📹 Kamera Langsung (Webcam)":
            if not is_model_available:
                st.warning("⚠️ File model YOLO belum ditemukan. Gunakan mode 'Unggah Foto' atau 'Ketik Manual'.")
            else:
                sub_cam = st.radio(
                    "Pilihan Tipe Kamera:",
                    ["⚡ Kamera Cepat (Snapshot Browser — Ringan & Anti Lag)", "📹 Kamera Streaming (WebRTC Real-Time)"],
                    horizontal=True,
                )

                # OPSI A: KAMERA CEPAT (NATIVE BROWSER HTML5 - 0% LAG)
                if "Snapshot" in sub_cam:
                    st.caption("💡 *Kamera browser langsung: sangat halus (60 FPS), tidak membebani CPU laptop, dan hasil foto sangat tajam.*")
                    cam_shot = st.camera_input("Arahkan kamera ke plat nomor kendaraan:", key="native_cam_input")
                    if cam_shot is not None:
                        file_bytes = np.asarray(bytearray(cam_shot.read()), dtype=np.uint8)
                        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                        with st.spinner("AI sedang membaca plat nomor dari foto..."):
                            from ocr_engine import extract_plate_text
                            model, reader = get_yolo_and_ocr(DEFAULT_MODEL_PATH)
                            preds = model.predict(source=img_cv, conf=conf_threshold, verbose=False, imgsz=640)
                            found = []
                            for p in preds:
                                for b in p.boxes:
                                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                                    crop = img_cv[max(0, y1):min(img_cv.shape[0], y2), max(0, x1):min(img_cv.shape[1], x2)]
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
                                st.session_state["current_source"] = "Kamera Browser"
                                st.toast(f"Plat terdeteksi: {best_t}", icon="✅")
                                st.rerun()
                            else:
                                total_boxes = sum(len(p.boxes) for p in preds) if preds else 0
                                if total_boxes == 0:
                                    st.error("❌ Tidak ada plat nomor kendaraan yang terdeteksi. AI mendeteksi ini bukan kendaraan (wajah / pakaian / benda acak).")
                                else:
                                    st.warning("⚠️ Kotak plat terdeteksi namun karakter tidak terbaca sebagai format plat nomor Indonesia yang sah.")

                # OPSI B: KAMERA STREAMING WEBRTC (OPTIMIZED 30 FPS)
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
                                self.box = None

                            def recv(self, frame):
                                img = frame.to_ndarray(format="bgr24")
                                self.frame_count += 1
                                h, w = img.shape[:2]

                                # Simpan frame asli untuk di-scan saat tombol ditekan
                                with self.lock:
                                    self.latest_frame = img.copy()

                                # Deteksi kotak plat dengan YOLO tiap N frame (Hanya YOLO ringan, TANPA EasyOCR di stream agar tidak lag)
                                if self.frame_count % frame_skip == 0:
                                    try:
                                        preds = model.predict(source=img, conf=conf_threshold, verbose=False, imgsz=320)
                                        found_box = None
                                        for p in preds:
                                            for b in p.boxes:
                                                found_box = list(map(int, b.xyxy[0].tolist()))
                                                break
                                            if found_box:
                                                break
                                        with self.lock:
                                            self.box = found_box
                                    except Exception:
                                        pass

                                # Tampilkan visual feedback ke layar
                                with self.lock:
                                    cur_box = self.box

                                if cur_box:
                                    x1, y1, x2, y2 = cur_box
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 0), 2)
                                    cv2.putText(img, "PLAT TERDETEKSI", (x1, max(20, y1 - 8)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2)
                                else:
                                    # Kotak panduan (viewfinder) di tengah
                                    cx, cy = w // 2, h // 2
                                    bx1, by1 = cx - 150, cy - 50
                                    bx2, by2 = cx + 150, cy + 50
                                    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 180, 255), 1, cv2.LINE_AA)
                                    cv2.putText(img, "Arahkan Plat ke Sini", (bx1 + 10, by1 - 8),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

                                return type(frame).from_ndarray(img, format="bgr24")

                        st.write("Arahkan kamera ke plat kendaraan:")
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
                            if snap is not None:
                                with st.spinner("AI sedang memeriksa plat nomor kendaraan..."):
                                    from ocr_engine import extract_plate_text
                                    preds = model.predict(source=snap, conf=conf_threshold, verbose=False, imgsz=640)
                                    found = []
                                    for p in preds:
                                        for b in p.boxes:
                                            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                                            crop = snap[max(0, y1):min(snap.shape[0], y2), max(0, x1):min(snap.shape[1], x2)]
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
                                        total_boxes = sum(len(p.boxes) for p in preds) if preds else 0
                                        if total_boxes == 0:
                                            st.error("❌ Tidak ada plat nomor kendaraan yang terdeteksi. AI mendeteksi ini bukan kendaraan (wajah / pakaian / benda acak).")
                                        else:
                                            st.warning("⚠️ Kotak terdeteksi namun karakter tidak terbaca sebagai format plat nomor Indonesia yang sah.")
                    except Exception as e:
                        st.error(f"Gagal mengaktifkan kamera streaming: {e}")

        # MODE 2: UNGGAH FOTO
        elif mode_input == "📁 Unggah Foto Kendaraan":
            st.write("Unggah foto motor atau mobil (kamera depan/belakang):")
            uploaded_file = st.file_uploader("Pilih file foto (JPG/PNG)", type=["jpg", "jpeg", "png"], key="gate_photo_upload")
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Foto Kendaraan", use_container_width=True)

                if st.button("🔍 Deteksi & Baca Plat Nomor", type="primary", use_container_width=True):
                    with st.spinner("AI menganalisis gambar kendaraan..."):
                        from ocr_engine import extract_plate_text
                        found = []
                        if is_model_available:
                            model, reader = get_yolo_and_ocr(DEFAULT_MODEL_PATH)
                            preds = model.predict(source=img_cv, conf=conf_threshold, verbose=False, imgsz=640)
                            for p in preds:
                                for b in p.boxes:
                                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                                    crop = img_cv[max(0, y1):min(img_cv.shape[0], y2), max(0, x1):min(img_cv.shape[1], x2)]
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
                            st.error("❌ Tidak dapat menemukan plat nomor kendaraan pada gambar ini. Pastikan foto memuat plat kendaraan bermotor (bukan wajah/benda acak).")

        # MODE 3: KETIK MANUAL
        else:
            st.write("Gunakan jika plat nomor kotor/rusak dan tidak terbaca oleh AI:")
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

    # PANEL KANAN: KARTU HASIL VERIFIKASI & KONFIRMASI PETUGAS
    with col_verify:
        st.markdown("#### 🎫 Hasil Verifikasi & Tiket Gerbang")

        active_plate = st.session_state.get("current_plate")

        if not active_plate:
            st.info("👋 Silakan scan kamera, upload foto, atau ketik plat di sebelah kiri untuk memulai pemeriksaan.")
        else:
            cleaned_plat = disambiguate_plate(active_plate)
            vehicle_record = db_manager.find_vehicle(cleaned_plat)
            
            # Cek status parkir aktif saat ini di database
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
            if dec_result["status_code"] == "INVALID_PLATE":
                badge_style = "status-invalid"
                status_title = "❌ BUKAN PLAT KENDARAAN (OBJEK TIDAK VALID)"
            elif dec_result["status_code"] == "OCR_UNCERTAIN":
                badge_style = "status-uncertain"
                status_title = "⚠️ TEKS BURAM / OCR KURANG YAKIN"
            elif dec_result["status_code"] == "REGISTERED":
                if dec_result["category"] == "Dosen":
                    badge_style = "status-dosen"
                    status_title = "DOSEN TERDAFTAR — PARKIR GRATIS"
                else:
                    badge_style = "status-mhs"
                    status_title = f"{dec_result['category'].upper()} TERDAFTAR — WAJIB BAYAR"
            else:
                badge_style = "status-tamu"
                status_title = "KENDARAAN TAMU (LUAR KAMPUS) — WAJIB BAYAR"

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

            # Tiket Verifikasi Digital
            card_html = (
                f'<div class="ticket-card">'
                f'<div style="display:flex; justify-content:space-between; align-items:flex-start;">'
                f'<div>'
                f'<div style="font-size:0.8rem; color:gray; font-weight:600; text-transform:uppercase;">Plat Nomor Terbaca</div>'
                f'<div class="plate-number-badge">{format_plate_display(cleaned_plat)}</div>'
                f'<div style="font-size:0.85rem; color:gray;">Sumber: <b>{st.session_state.get("current_source")}</b> (OCR Conf: {st.session_state.get("current_conf", 1.0):.0%})</div>'
                f'</div>'
                f'<div>'
                f'<span class="badge-status {badge_style}">{status_title}</span>'
                f'</div>'
                f'</div>'
                f'<hr style="margin: 16px 0; border: none; border-top: 1px solid rgba(128,128,128,0.2);" />'
                f'<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 14px; font-size: 0.95rem;">'
                f'<div>👤 <b>Nama Pemilik:</b><br><span style="font-size:1.1rem; font-weight:700;">{dec_result.get("owner_name") or "Tidak Valid"}</span></div>'
                f'<div>🏷️ <b>Kategori:</b><br><span style="font-size:1.1rem; font-weight:700;">{dec_result.get("category")}</span></div>'
                f'<div>💵 <b>Tarif Parkir:</b><br><span style="font-size:1.4rem; font-weight:800; color:#2563eb;">{rupiah(dec_result.get("fee"))}</span></div>'
                f'<div>🚪 <b>Status Gerbang:</b><br><span style="font-size:1.05rem; font-weight:700;">{gate_status_label}</span></div>'
                f'{extra_gate_info}'
                f'</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

            if dec_result["status_code"] == "INVALID_PLATE":
                st.error("🚫 **Objek Tidak Sah:** AI mendeteksi bahwa objek ini bukan plat nomor kendaraan bermotor yang sah (wajah, pakaian, atau teks acak). Gerbang tidak dapat dibuka.")
                if st.button("🔄 Reset / Periksa Ulang", type="secondary", use_container_width=True):
                    st.session_state["current_plate"] = None
                    st.rerun()
            else:
                # Cek Peringatan Duplikasi
                recent = db_manager.check_recent_transaction(cleaned_plat, cooldown_seconds=cooldown_sec, gate_direction=gate_code)
                if recent:
                    st.warning(f"⚠️ Peringatan: Kendaraan {format_plate_display(cleaned_plat)} sudah tercatat pada {recent['created_at']} (arah {gate_code}).")

                # Form Konfirmasi Petugas
                with st.form("form_konfirmasi_gerbang"):
                    koreksi_plat = st.text_input("Koreksi Plat Nomor (Jika Salah Baca)", value=format_plate_display(cleaned_plat))
                    catatan_petugas = st.text_input("Catatan Petugas (Opsional)", placeholder="Contoh: Mengantar dokumen ke Rektorat")
                    
                    btn_label = "🚪 Buka Gerbang Masuk (Simpan Transaksi)" if gate_code == "IN" else "🚪 Buka Gerbang Keluar & Selesai Parkir"
                    btn_simpan_tx = st.form_submit_button(btn_label, type="primary", use_container_width=True)

                if btn_simpan_tx:
                    final_p = koreksi_plat.strip() if koreksi_plat.strip() else cleaned_plat
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

                    tx_id = db_manager.save_transaction(
                        plate=final_p,
                        owner_name=final_dec.get("owner_name"),
                        category=final_dec.get("category"),
                        fee=final_dec.get("fee"),
                        decision=final_dec.get("decision"),
                        status="CONFIRMED",
                        gate_direction=final_gate_code,
                        notes=catatan_petugas,
                        duration_minutes=final_dec.get("duration_minutes", 0),
                        entry_time=final_dec.get("entry_time"),
                    )
                    st.success(f"🎉 Transaksi #{tx_id} berhasil dicatat! Gerbang {final_gate_code} Terbuka.")
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
    st.subheader("Arsitektur Sistem & Status Model AI")

    st.markdown("""
### 🔄 Cara Kerja Sistem (Pipeline)
1. **YOLO (Deteksi Plat):** Menemukan posisi kotak plat nomor pada foto/video kendaraan.
2. **EasyOCR (Pembacaan Karakter):** Memotong kotak plat, membersihkan gambar (CLAHE contrast), dan membaca teks alfanumerik.
3. **Pencocokan Database SQLite:** Teks plat dicocokkan ke daftar registrasi kampus.
   - **Dosen:** Tarif **Rp0 (Gratis)**.
   - **Mahasiswa:** Tarif **Rp2.000**.
   - **Tamu / Luar Kampus:** Tarif **Rp5.000**.
4. **Konfirmasi Petugas:** Petugas memverifikasi fisik kendaraan dan mencatat transaksi resmi ke sistem.
    """)

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
