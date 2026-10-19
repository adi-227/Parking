import streamlit as st
from typing import Dict, Optional


def render_panduan_ai(status: Optional[Dict[str, bool]] = None):
    """
    Komponen Visual Panduan & Status AI untuk Smart Parking Kampus (Dark Theme).
    Menjelaskan pipeline deteksi plat nomor dari awal hingga akhir dengan
    tampilan modern, profesional, dan siap dipresentasikan.
    """
    if status is None:
        status = {
            "yolo": True,
            "ocr": True,
            "database": True,
            "camera": True,
        }

    # Helper status boolean to class & label
    yolo_active = status.get("yolo", True)
    ocr_active = status.get("ocr", True)
    db_active = status.get("database", True)
    cam_active = status.get("camera", True)

    html_code = f"""
<style>
/* ==============================================================================
   PANDUAN & STATUS AI - DESIGN SYSTEM (DARK THEME)
   ============================================================================== */
:root {{
    --guide-bg: #0E1117;
    --guide-card-bg: #161B26;
    --guide-card-border: rgba(255, 255, 255, 0.08);
    --guide-accent-blue: #2563EB;
    --guide-hover-blue: #1D4ED8;
    --guide-text-primary: #F3F4F6;
    --guide-text-secondary: #8B93A7;
    --guide-success-green: #4ADE80;
    --guide-success-bg: rgba(31, 107, 58, 0.35);
    --guide-gold: #C9A96E;
    --guide-gold-bg: rgba(201, 169, 110, 0.18);
    --guide-radius: 14px;
}}

.guide-wrapper {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: var(--guide-text-primary);
    max-width: 100%;
    margin: 0 auto;
    padding: 4px 0 24px 0;
}}

/* ------------------------------------------------------------------------------
   1. PANEL STATUS SISTEM AI (PULSING INDICATORS)
   ------------------------------------------------------------------------------ */
.ai-status-panel {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}}

.status-chip-card {{
    background: var(--guide-card-bg);
    border: 1px solid var(--guide-card-border);
    border-radius: var(--guide-radius);
    padding: 12px 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}

.status-chip-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.16);
    border-color: rgba(37, 99, 235, 0.4);
}}

.status-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
}}

.status-dot.active {{
    background: var(--guide-success-green);
    box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7);
    animation: statusPulse 2s infinite;
}}

.status-dot.inactive {{
    background: #EF4444;
    box-shadow: 0 0 6px rgba(239, 68, 68, 0.5);
}}

@keyframes statusPulse {{
    0% {{
        box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7);
    }}
    70% {{
        box-shadow: 0 0 0 8px rgba(74, 222, 128, 0);
    }}
    100% {{
        box-shadow: 0 0 0 0 rgba(74, 222, 128, 0);
    }}
}}

.status-text-group {{
    display: flex;
    flex-direction: column;
}}

.status-title {{
    font-size: 0.84rem;
    font-weight: 700;
    color: #F9FAFB;
}}

.status-meta {{
    font-size: 0.70rem;
    color: var(--guide-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}}

/* ------------------------------------------------------------------------------
   2. HERO BANNER
   ------------------------------------------------------------------------------ */
.guide-hero {{
    background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 50%, #2563EB 100%);
    border-radius: 16px;
    padding: 26px 28px;
    margin-bottom: 30px;
    box-shadow: 0 10px 28px rgba(30, 58, 138, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.15);
}}

.guide-hero-title {{
    font-size: 1.55rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 8px 0;
    letter-spacing: -0.01em;
}}

.guide-hero-subtitle {{
    font-size: 0.92rem;
    color: #DBEAFE;
    margin: 0 0 18px 0;
    line-height: 1.5;
    max-width: 820px;
}}

.guide-hero-badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}}

.hero-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 0.02em;
}}

/* ------------------------------------------------------------------------------
   3. PIPELINE TIMELINE VERTIKAL 7 TAHAP
   ------------------------------------------------------------------------------ */
.section-heading {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 18px 0;
}}

.section-heading h3 {{
    font-size: 1.18rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0;
}}

.timeline-container {{
    position: relative;
    padding-left: 24px;
    margin-top: 10px;
    margin-bottom: 32px;
}}

.timeline-line {{
    position: absolute;
    left: 45px;
    top: 24px;
    bottom: 24px;
    width: 3px;
    background: linear-gradient(180deg, #2563EB 0%, #3B82F6 40%, #10B981 85%, #059669 100%);
    border-radius: 2px;
    z-index: 1;
}}

.timeline-step {{
    position: relative;
    display: flex;
    gap: 20px;
    margin-bottom: 20px;
    z-index: 2;
}}

.step-circle {{
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
    border: 3px solid #0E1117;
    box-shadow: 0 0 12px rgba(37, 99, 235, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    font-weight: 800;
    color: #FFFFFF;
    flex-shrink: 0;
}}

.timeline-card {{
    flex: 1;
    background: var(--guide-card-bg);
    border: 1px solid var(--guide-card-border);
    border-radius: var(--guide-radius);
    padding: 16px 20px;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}}

.timeline-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.2);
    border-color: rgba(37, 99, 235, 0.45);
}}

.timeline-card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 8px;
}}

.timeline-card-title {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.98rem;
    font-weight: 700;
    color: #FFFFFF;
}}

.tech-chip-group {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}}

.tech-chip {{
    background: rgba(37, 99, 235, 0.16);
    color: #93C5FD;
    border: 1px solid rgba(37, 99, 235, 0.35);
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    font-family: Consolas, monospace;
}}

.timeline-card-desc {{
    font-size: 0.86rem;
    color: #D1D5DB;
    line-height: 1.55;
    margin: 0;
}}

/* ------------------------------------------------------------------------------
   4. CONTOH KOREKSI OCR (DISAMBIGUASI)
   ------------------------------------------------------------------------------ */
.ocr-demo-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-bottom: 30px;
}}

.ocr-demo-card {{
    background: var(--guide-card-bg);
    border: 1px solid var(--guide-card-border);
    border-radius: var(--guide-radius);
    padding: 16px 20px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}

.ocr-demo-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    border-color: rgba(37, 99, 235, 0.35);
}}

.ocr-demo-title {{
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--guide-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}}

.ocr-comparison-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    background: #0E1117;
    border-radius: 10px;
    padding: 12px 14px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}}

.ocr-plate-box {{
    display: flex;
    flex-direction: column;
    gap: 4px;
}}

.ocr-label-small {{
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #9CA3AF;
}}

.plate-raw {{
    font-family: Consolas, monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #F87171;
    letter-spacing: 0.04em;
}}

.plate-corrected {{
    font-family: Consolas, monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #4ADE80;
    letter-spacing: 0.04em;
}}

.hl-yellow {{
    background: rgba(251, 191, 36, 0.25);
    color: #FCD34D;
    padding: 0 3px;
    border-radius: 4px;
    border-bottom: 2px solid #F59E0B;
}}

.hl-green {{
    background: rgba(74, 222, 128, 0.25);
    color: #86EFAC;
    padding: 0 3px;
    border-radius: 4px;
    border-bottom: 2px solid #10B981;
}}

.arrow-icon {{
    font-size: 1.2rem;
    color: var(--guide-text-secondary);
}}

.ocr-explanation {{
    font-size: 0.78rem;
    color: #9CA3AF;
    margin-top: 10px;
    line-height: 1.4;
}}

/* ------------------------------------------------------------------------------
   5. KARTU TARIF SIVITAS & TAMU
   ------------------------------------------------------------------------------ */
.tariff-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 30px;
}}

.tariff-card {{
    background: var(--guide-card-bg);
    border: 1px solid var(--guide-card-border);
    border-radius: var(--guide-radius);
    padding: 20px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    position: relative;
    overflow: hidden;
}}

.tariff-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
}}

.tariff-card.dosen::before {{
    background: linear-gradient(90deg, #10B981, #34D399);
}}

.tariff-card.mahasiswa::before {{
    background: linear-gradient(90deg, #2563EB, #60A5FA);
}}

.tariff-card.tamu::before {{
    background: linear-gradient(90deg, #D97706, #FBBF24);
}}

.tariff-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}}

.tariff-badge {{
    display: inline-block;
    align-self: flex-start;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
}}

.tariff-badge.green {{
    background: var(--guide-success-bg);
    color: var(--guide-success-green);
    border: 1px solid rgba(74, 222, 128, 0.3);
}}

.tariff-badge.blue {{
    background: rgba(37, 99, 235, 0.2);
    color: #93C5FD;
    border: 1px solid rgba(37, 99, 235, 0.35);
}}

.tariff-badge.gold {{
    background: var(--guide-gold-bg);
    color: var(--guide-gold);
    border: 1px solid rgba(201, 169, 110, 0.35);
}}

.tariff-role {{
    font-size: 1.05rem;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 4px;
}}

.tariff-price {{
    font-size: 1.8rem;
    font-weight: 800;
    margin: 8px 0;
    letter-spacing: -0.02em;
}}

.tariff-price.free {{
    color: var(--guide-success-green);
}}

.tariff-price.mhs {{
    color: #60A5FA;
}}

.tariff-price.guest {{
    color: #FBBF24;
}}

.tariff-desc {{
    font-size: 0.78rem;
    color: var(--guide-text-secondary);
    line-height: 1.45;
    margin: 0;
}}

/* ------------------------------------------------------------------------------
   6. TABEL TECH STACK SISTEM
   ------------------------------------------------------------------------------ */
.table-card {{
    background: var(--guide-card-bg);
    border: 1px solid var(--guide-card-border);
    border-radius: var(--guide-radius);
    padding: 6px;
    overflow-x: auto;
    margin-bottom: 16px;
}}

.tech-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    text-align: left;
}}

.tech-table th {{
    background: #0E1117;
    color: #9CA3AF;
    font-weight: 700;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}}

.tech-table td {{
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    color: #E5E7EB;
    vertical-align: middle;
}}

.tech-table tr:last-child td {{
    border-bottom: none;
}}

.tech-table tr:nth-child(even) {{
    background: rgba(255, 255, 255, 0.015);
}}

.tech-table tr:hover {{
    background: rgba(37, 99, 235, 0.06);
}}

.table-tech-chip {{
    display: inline-block;
    background: rgba(37, 99, 235, 0.18);
    border: 1px solid rgba(37, 99, 235, 0.35);
    color: #93C5FD;
    padding: 3px 10px;
    border-radius: 6px;
    font-family: Consolas, monospace;
    font-size: 0.78rem;
    font-weight: 700;
}}

/* ------------------------------------------------------------------------------
   RESPONSIVE & ACCESSIBILITY
   ------------------------------------------------------------------------------ */
@media (max-width: 768px) {{
    .ai-status-panel {{
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
    }}
    .guide-hero {{
        padding: 20px 18px;
    }}
    .guide-hero-title {{
        font-size: 1.3rem;
    }}
    .timeline-line {{
        left: 36px;
    }}
    .step-circle {{
        width: 38px;
        height: 38px;
        font-size: 0.85rem;
    }}
    .timeline-container {{
        padding-left: 10px;
    }}
    .timeline-step {{
        gap: 12px;
    }}
    .ocr-demo-grid {{
        grid-template-columns: 1fr;
    }}
    .tariff-grid {{
        grid-template-columns: 1fr;
    }}
    .tech-table th, .tech-table td {{
        padding: 10px 12px;
    }}
}}

@media (prefers-reduced-motion: reduce) {{
    .status-dot.active {{
        animation: none;
    }}
    .status-chip-card, .timeline-card, .ocr-demo-card, .tariff-card {{
        transition: none !important;
        transform: none !important;
    }}
}}
</style>

<div class="guide-wrapper">

    <!-- 1. PANEL STATUS SISTEM AI -->
    <div class="ai-status-panel">
        <div class="status-chip-card">
            <span class="status-dot {'active' if yolo_active else 'inactive'}"></span>
            <div class="status-text-group">
                <span class="status-title">Model YOLOv8</span>
                <span class="status-meta">{'Aktif & Siap' if yolo_active else 'Nonaktif'}</span>
            </div>
        </div>
        <div class="status-chip-card">
            <span class="status-dot {'active' if ocr_active else 'inactive'}"></span>
            <div class="status-text-group">
                <span class="status-title">EasyOCR Engine</span>
                <span class="status-meta">{'Siap Deteksi' if ocr_active else 'Nonaktif'}</span>
            </div>
        </div>
        <div class="status-chip-card">
            <span class="status-dot {'active' if db_active else 'inactive'}"></span>
            <div class="status-text-group">
                <span class="status-title">Database SQLite</span>
                <span class="status-meta">{'Terhubung' if db_active else 'Terputus'}</span>
            </div>
        </div>
        <div class="status-chip-card">
            <span class="status-dot {'active' if cam_active else 'inactive'}"></span>
            <div class="status-text-group">
                <span class="status-title">Kamera WebRTC</span>
                <span class="status-meta">{'Online / Standby' if cam_active else 'Offline'}</span>
            </div>
        </div>
    </div>

    <!-- 2. HERO BANNER -->
    <div class="guide-hero">
        <h2 class="guide-hero-title">Alur Kerja Sistem Smart Parking</h2>
        <p class="guide-hero-subtitle">
            Arsitektur pipeline berbasis Computer Vision dan Deep Learning untuk mengenali plat nomor kendaraan (mobil & motor) secara realtime dari berbagai sudut, mencocokkan registrasi kampus, serta mengontrol palang otomatis.
        </p>
        <div class="guide-hero-badges">
            <span class="hero-pill">🎯 Ultralytics YOLO</span>
            <span class="hero-pill">🔍 EasyOCR + PyTorch</span>
            <span class="hero-pill">📹 Streamlit WebRTC</span>
            <span class="hero-pill">🗄️ SQLite3 Database</span>
        </div>
    </div>

    <!-- 3. PIPELINE TIMELINE VERTIKAL 7 TAHAP -->
    <div class="section-heading">
        <h3>🔄 Pipeline Pemrosesan Citra & Logika Parkir (7 Tahap)</h3>
    </div>

    <div class="timeline-container">
        <div class="timeline-line"></div>

        <!-- TAHAP 1 -->
        <div class="timeline-step">
            <div class="step-circle">1</div>
            <div class="timeline-card">
                <div class="timeline-card-header">
                    <div class="timeline-card-title">
                        <span>📹 Input Citra (Capture & Live Streaming)</span>
                    </div>
                    <div class="tech-chip-group">
                        <span class="tech-chip">WebRTC</span>
                        <span class="tech-chip">Zero-Install</span>
                        <span class="tech-chip">Mobile & PC</span>
                    </div>
                </div>
                <p class="timeline-card-desc">
                    Video real-time dialirkan dari kamera laptop maupun smartphone petugas pos menggunakan protokol <strong>WebRTC (streamlit-webrtc)</strong>. Petugas cukup mengakses URL peramban (browser) secara aman tanpa perlu instalasi aplikasi khusus.
                </p>
            </div>
        </div>

        <!-- TAHAP 2 -->
        <div class="timeline-step">
            <div class="step-circle">2</div>
            <div class="timeline-card">
                <div class="timeline-card-header">
                    <div class="timeline-card-title">
                        <span>🎯 Deteksi Lokasi Plat Nomor (YOLO)</span>
                    </div>
                    <div class="tech-chip-group">
                        <span class="tech-chip">YOLOv8</span>
                        <span class="tech-chip">AR 1.10 - 5.80</span>
                        <span class="tech-chip">Motor 52% Split</span>
                    </div>
                </div>
                <p class="timeline-card-desc">
                    Model AI melokalisasi kotak plat nomor pada resolusi dinamis. Dilengkapi <strong>filter aspek rasio (1.10 – 5.80)</strong> untuk menolak objek bukan plat seperti wajah, telapak tangan, atau sablon baju. Untuk plat motor yang menyatu dengan spatbor belakang, AI otomatis memotong ke <strong>52% area atas</strong> tempat plat berada.
                </p>
            </div>
        </div>

        <!-- TAHAP 3 -->
        <div class="timeline-step">
            <div class="step-circle">3</div>
            <div class="timeline-card">
                <div class="timeline-card-header">
                    <div class="timeline-card-title">
                        <span>🟩 Stabilisasi Bounding Box (SmoothBoxTracker)</span>
                    </div>
                    <div class="tech-chip-group">
                        <span class="tech-chip">EMA α=0.65</span>
                        <span class="tech-chip">IoU Matching</span>
                        <span class="tech-chip">Grace Period 8-Frames</span>
                    </div>
                </div>
                <p class="timeline-card-desc">
                    Pelacak temporal menghilangkan jitter (getaran tangan) menggunakan <strong>Exponential Moving Average (EMA)</strong>. Kotak deteksi dikunci kontinuitasnya dengan <strong>IoU Proximity Matching</strong>, serta diberi <em>grace period</em> (8 frame) agar kotak hijau tidak berkedip atau hilang saat kamera bergerak.
                </p>
            </div>
        </div>

        <!-- TAHAP 4 -->
        <div class="timeline-step">
            <div class="step-circle">4</div>
            <div class="timeline-card">
                <div class="timeline-card-header">
                    <div class="timeline-card-title">
                        <span>📐 Pelurusan Kemiringan Plat (Auto-Deskewing)</span>
                    </div>
                    <div class="tech-chip-group">
                        <span class="tech-chip">OpenCV</span>
                        <span class="tech-chip">Hough Line</span>
                        <span class="tech-chip">Angle ±45°</span>
                    </div>
                </div>
                <p class="timeline-card-desc">
                    Posisi kamera pos parkir umumnya berada di samping kendaraan. Algoritma <strong>Hough Line Transform</strong> mendeteksi sudut kemiringan plat hingga sudut <strong>45°</strong> (serong kanan/kiri), kemudian memutar (rotasi) citra plat secara otomatis agar lurus horizontal sebelum dibaca OCR.
                </p>
            </div>
        </div>

        <!-- TAHAP 5 -->
        <div class="timeline-step">
            <div class="step-circle">5</div>
            <div class="timeline-card">
                <div class="timeline-card-header">
                    <div class="timeline-card-title">
                        <span>🔍 Pembacaan Karakter (EasyOCR + Heuristik Plat)</span>
                    </div>
                    <div class="tech-chip-group">
                        <span class="tech-chip">EasyOCR</span>
                        <span class="tech-chip">CLAHE</span>
                        <span class="tech-chip">Indonesian Regex</span>
                    </div>
                </div>
                <p class="timeline-card-desc">
                    Citra plat ditingkatkan kontrasnya dengan CLAHE. <strong>EasyOCR</strong> mengekstrak teks karakter alfanumerik. Modul disambiguasi konteks plat Indonesia <code>[1-2 Wilayah][1-4 Angka][0-3 Seri]</code> mengoreksi karakter tertukar otomatis (misal: <em>O/D &rarr; 0</em>, <em>I &rarr; 1</em> pada angka tengah, serta <em>4 &rarr; A</em> pada seri).
                </p>
            </div>
        </div>

        <!-- TAHAP 6 -->
        <div class="timeline-step">
            <div class="step-circle">6</div>
            <div class="timeline-card">
                <div class="timeline-card-header">
                    <div class="timeline-card-title">
                        <span>💳 Logika Parkir & Skema Tarif (parking.db)</span>
                    </div>
                    <div class="tech-chip-group">
                        <span class="tech-chip">SQLite3</span>
                        <span class="tech-chip">Role-Based Fee</span>
                        <span class="tech-chip">Duration Calc</span>
                    </div>
                </div>
                <p class="timeline-card-desc">
                    Plat dicocokkan ke database kampus: <strong>Dosen (Rp0 / Gratis)</strong>, <strong>Mahasiswa (Rp2.000 flat)</strong>, dan <strong>Tamu (Rp5.000)</strong>. Pada gerbang KELUAR (OUT), sistem otomatis menghitung durasi parkir dari waktu masuk terakhir (contoh: <em>45 Menit</em> atau <em>2 Jam 10 Menit</em>).
                </p>
            </div>
        </div>

        <!-- TAHAP 7 -->
        <div class="timeline-step">
            <div class="step-circle">7</div>
            <div class="timeline-card">
                <div class="timeline-card-header">
                    <div class="timeline-card-title">
                        <span>🚧 Kontrol Palang & Pencatatan Transaksi</span>
                    </div>
                    <div class="tech-chip-group">
                        <span class="tech-chip">Barrier Actuator</span>
                        <span class="tech-chip">Audit Trail</span>
                        <span class="tech-chip">Live Analytics</span>
                    </div>
                </div>
                <p class="timeline-card-desc">
                    Status palang gerbang berubah menjadi <strong>BUKA</strong> secara visual dengan indikator animasi. Seluruh data transaksi (waktu, plat, nama pemilik, kategori, durasi, dan biaya) dicatat permanen ke database serta disinkronkan ke <strong>Dashboard Ringkasan</strong>.
                </p>
            </div>
        </div>
    </div>

    <!-- 4. VISUAL CONTOH KOREKSI OCR -->
    <div class="section-heading">
        <h3>🧠 Contoh Disambiguasi Cerdas Karakter Plat Indonesia</h3>
    </div>

    <div class="ocr-demo-grid">
        <div class="ocr-demo-card">
            <div class="ocr-demo-title">🚗 Contoh 1: Kasus Plat Mobil (Huruf O terbaca di tengah angka)</div>
            <div class="ocr-comparison-row">
                <div class="ocr-plate-box">
                    <span class="ocr-label-small">Mentah dari OCR</span>
                    <span class="plate-raw">B 6<span class="hl-yellow">O</span>91 JGU</span>
                </div>
                <span class="arrow-icon">➔</span>
                <div class="ocr-plate-box">
                    <span class="ocr-label-small">Hasil Disambiguasi AI</span>
                    <span class="plate-corrected">B 6<span class="hl-green">0</span>91 JGU</span>
                </div>
            </div>
            <p class="ocr-explanation">
                Aturan tata bahasa plat Indonesia mewajibkan segmen tengah adalah angka nomor polisi. AI secara otomatis mengubah huruf 'O' menjadi angka '0' dan merapikan spasi standar.
            </p>
        </div>

        <div class="ocr-demo-card">
            <div class="ocr-demo-title">🏍️ Contoh 2: Kasus Plat Motor (Noise Border & Angka 4)</div>
            <div class="ocr-comparison-row">
                <div class="ocr-plate-box">
                    <span class="ocr-label-small">Mentah dari OCR</span>
                    <span class="plate-raw">F<span class="hl-yellow">L</span>220 FG<span class="hl-yellow">H</span></span>
                </div>
                <span class="arrow-icon">➔</span>
                <div class="ocr-plate-box">
                    <span class="ocr-label-small">Hasil Disambiguasi AI</span>
                    <span class="plate-corrected">F <span class="hl-green">4</span>220 FG<span class="hl-green">M</span></span>
                </div>
            </div>
            <p class="ocr-explanation">
                Huruf 'L' yang tertukar dengan angka '4' dan noise baut belakang pada huruf 'M' dikoreksi berdasarkan perbandingan kamus seri dan pola wilayah nopol Bogor/Jawa Barat.
            </p>
        </div>
    </div>

    <!-- 5. KARTU TARIF -->
    <div class="section-heading">
        <h3>🎫 Kebijakan Tarif & Hak Akses Parkir Kampus</h3>
    </div>

    <div class="tariff-grid">
        <div class="tariff-card dosen">
            <div>
                <span class="tariff-badge green">SIVITAS AKADEMIKA</span>
                <div class="tariff-role">👨‍🏫 Dosen & Tendik</div>
                <div class="tariff-price free">Rp 0</div>
            </div>
            <p class="tariff-desc">
                Akses parkir gratis khusus dosen, staf, dan tenaga kependidikan tetap/luar biasa yang terdaftar resmi di basis data kampus.
            </p>
        </div>

        <div class="tariff-card mahasiswa">
            <div>
                <span class="tariff-badge blue">SUBSIDI KAMPUS</span>
                <div class="tariff-role">🎓 Mahasiswa Aktif</div>
                <div class="tariff-price mhs">Rp 2.000</div>
            </div>
            <p class="tariff-desc">
                Tarif subsidi flat sekali masuk untuk mahasiswa aktif terdaftar. Tidak dikenakan biaya per jam tambahan.
            </p>
        </div>

        <div class="tariff-card tamu">
            <div>
                <span class="tariff-badge gold">PENGUNJUNG LUAR</span>
                <div class="tariff-role">🚗 Tamu / Umum</div>
                <div class="tariff-price guest">Rp 5.000</div>
            </div>
            <p class="tariff-desc">
                Kendaraan tamu atau luar kampus yang belum terdaftar. Tiket virtual otomatis diterbitkan saat masuk melalui gerbang IN.
            </p>
        </div>
    </div>

    <!-- 6. TABEL TECH STACK -->
    <div class="section-heading">
        <h3>💻 Arsitektur Teknologi (Tech Stack)</h3>
    </div>

    <div class="table-card">
        <table class="tech-table">
            <thead>
                <tr>
                    <th style="width: 25%;">Komponen</th>
                    <th style="width: 30%;">Teknologi</th>
                    <th style="width: 45%;">Peran & Fungsionalitas</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>User Interface</strong></td>
                    <td><span class="table-tech-chip">Streamlit (Python)</span></td>
                    <td>Antarmuka web modern, responsif desktop & smartphone petugas</td>
                </tr>
                <tr>
                    <td><strong>Live Stream</strong></td>
                    <td><span class="table-tech-chip">Streamlit WebRTC</span></td>
                    <td>Streaming video kamera pos/HP berlatensi rendah langsung lewat browser</td>
                </tr>
                <tr>
                    <td><strong>Object Detection</strong></td>
                    <td><span class="table-tech-chip">Ultralytics YOLOv8</span></td>
                    <td>Lokalisasi bounding box plat nomor secara presisi dan cepat</td>
                </tr>
                <tr>
                    <td><strong>Image Processing</strong></td>
                    <td><span class="table-tech-chip">OpenCV (cv2)</span></td>
                    <td>Auto-deskewing (Hough), CLAHE contrast, dan filtering rasio aspek</td>
                </tr>
                <tr>
                    <td><strong>OCR Engine</strong></td>
                    <td><span class="table-tech-chip">EasyOCR + PyTorch</span></td>
                    <td>Pengenalan karakter alfanumerik pada plat nomor kendaraan</td>
                </tr>
                <tr>
                    <td><strong>Tracking Filter</strong></td>
                    <td><span class="table-tech-chip">EMA + IoU Matching</span></td>
                    <td>Anti-jitter, pencegah flicker, dan stabilisasi pelacak gerak kamera</td>
                </tr>
                <tr>
                    <td><strong>Database</strong></td>
                    <td><span class="table-tech-chip">SQLite3 (parking.db)</span></td>
                    <td>Penyimpanan data master sivitas, tarif, dan audit log transaksi</td>
                </tr>
                <tr>
                    <td><strong>Tunneling / PWA</strong></td>
                    <td><span class="table-tech-chip">Localtunnel</span></td>
                    <td>Membuka sambungan HTTPS kamera HP nirkabel secara aman</td>
                </tr>
            </tbody>
        </table>
    </div>

</div>
"""
    st.markdown(html_code, unsafe_allow_html=True)

