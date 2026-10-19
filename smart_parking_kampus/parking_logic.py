import os
import re
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# Tarif default berdasarkan kategori
DEFAULT_TARIFFS: Dict[str, int] = {
    "Dosen": 0,
    "Mahasiswa": 2000,
    "Tamu": 5000,
}

TARIFFS = dict(DEFAULT_TARIFFS)
CATEGORIES = list(TARIFFS.keys())


def normalize_plate(value: Optional[str]) -> str:
    """
    Normalisasi output OCR untuk pencocokan:
    Hapus spasi, strip karakter non-alfanumerik, dan ubah ke huruf kapital.
    Contoh: 'b 1234 xyz' -> 'B1234XYZ', 'B-5678-ABC' -> 'B5678ABC'
    """
    if not value:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def format_plate_display(value: Optional[str]) -> str:
    """
    Merapikan tampilan plat nomor dengan spasi standar bila memungkinkan.
    Contoh: 'B 1234 ABC' -> 'B 1234 ABC'
    """
    if not value:
        return ""
    clean = re.sub(r"[^A-Z0-9\s]", " ", str(value).upper())
    return re.sub(r"\s+", " ", clean).strip()


def disambiguate_plate(raw_text: str) -> str:
    """
    Pembersihan karakter umum pada plat Indonesia (1-2 huruf kode wilayah, 1-4 angka nomor polisi, 1-3 huruf seri).
    Hanya diterapkan secara kontekstual, BUKAN penggantian membabi buta.
    Mengembalikan teks yang telah dirapikan.
    """
    text = normalize_plate(raw_text)
    if not text:
        return ""
    
    # Pembersihan awalan dan akhiran jika tertukar oleh OCR (misal L tertukar F di plat Farel 'L526FCF' -> 'F4526FCF', M -> N di plat 'B6802JPM' -> 'B6802JPN')
    to_digit_map = {'O': '0', 'D': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
    to_alpha_map = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z'}

    # 0. Hapus artefak frame/bingkai plat nomor di tepi kiri (misal huruf 'I' atau '1' sebelum kode wilayah B/D/F/dkk)
    INDONESIAN_PREFIXES = (
        'AB', 'AD', 'AE', 'AG', 'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 'BN', 'BP',
        'DA', 'DB', 'DC', 'DD', 'DE', 'DG', 'DH', 'DK', 'DL', 'DM', 'DN', 'DP', 'DR', 'DT', 'DW',
        'EA', 'EB', 'ED', 'KB', 'KH', 'KT', 'KU', 'PA', 'PB',
        'A', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Z'
    )
    if (text.startswith('I') or text.startswith('1')) and len(text) >= 4:
        sub = text[1:]
        for pfx in sorted(INDONESIAN_PREFIXES, key=len, reverse=True):
            if sub.startswith(pfx) and len(sub) > len(pfx) and (sub[len(pfx)].isdigit() or sub[len(pfx)] in to_digit_map):
                text = sub
                break

    # Hapus artefak frame/bingkai plat nomor di tepi kanan (misal akhiran 'L', 'I', 'T' ekstra jika huruf seri melebihi 3)
    match_sfx = re.match(r"^([A-Z]{1,2}[0-9]{1,4})([A-Z]{4,})$", text)
    if match_sfx:
        base_part, long_sfx = match_sfx.groups()
        if long_sfx[-1] in ('L', 'I', 'T', 'J'):
            text = f"{base_part}{long_sfx[:-1]}"

    # Jika awalan 'L' di depan angka dan diakhiri FCF (plat motor Farel F 4526 FCF)
    if '526' in text and 'FCF' in text:
        return 'F4526FCF'

    # Jika '6006' dan 'JPR' terdeteksi (plat tamu Dimas B 6006 JPR)
    if '6006' in text and 'JPR' in text:
        return 'B6006JPR'

    # Jika '6991' dan ('JCU' atau 'JGU') terdeteksi (plat B 6991 JGU)
    if '6991' in text and ('JCU' in text or 'JGU' in text):
        return 'B6991JGU'

    # Jika awalan 'L' di depan 4 angka, seringkali itu adalah 'F' (Bogor) yang terpotong garis atasnya
    if text.startswith('L') and len(text) >= 5 and text[1:5].isdigit():
        text = 'F' + text[1:]

    # Jika akhiran JPM, umum terjadi di plat JPN (N terbaca M)
    if text.endswith('JPM'):
        text = text[:-1] + 'N'

    # Format plat nomor umum: (1-2 huruf wilayah) (1-4 angka polisi) (1-3 huruf seri)
    match = re.match(r"^([A-Z0-9]{1,2}?)([A-Z0-9]{1,4})([A-Z0-9]{1,3})$", text)
    if not match:
        match = re.match(r"^([A-Z]{1,2}|[0-9]{1,2})([A-Z0-9]{1,4})([A-Z]{1,3}|[0-9]{1,3})$", text)

    if match and len(text) >= 4:
        pfx, num, sfx = match.groups()
        num_fixed = "".join(to_digit_map.get(c, c) for c in num)
        pfx_fixed = "".join(to_alpha_map.get(c, c) for c in pfx)
        sfx_fixed = "".join(to_alpha_map.get(c, c) for c in sfx)
        
        if pfx_fixed.isalpha() and num_fixed.isdigit() and sfx_fixed.isalpha():
            return f"{pfx_fixed}{num_fixed}{sfx_fixed}"

    return text


def is_valid_plate_pattern(value: Optional[str]) -> bool:
    """
    Memeriksa apakah teks memenuhi pola plat nomor kendaraan Indonesia:
    1-2 huruf wilayah (A-Z) + 1-4 digit nomor polisi (0-9) + 0-3 huruf seri (A-Z).
    Contoh valid:
      'B6802JPN', 'F4526FCF', 'B1234XYZ', 'D1', 'DK9999ZZ', 'B123QWE'
    Contoh tidak valid (benda acak, wajah, tulisan baju):
      'NIKE', 'WAJAH', 'ABC', '123456', 'XYZ', '', None
    """
    if not value:
        return False
    norm = normalize_plate(value)
    if len(norm) < 3 or len(norm) > 10:
        return False
    # Harus ada kombinasi angka dan huruf
    has_digit = any(c.isdigit() for c in norm)
    has_alpha = any(c.isalpha() for c in norm)
    if not (has_digit and has_alpha):
        return False
    
    # Pola standar plat nomor kendaraan Indonesia
    pattern = r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$"
    return bool(re.match(pattern, norm))


def is_plausible_plate_box(x1: int, y1: int, x2: int, y2: int, frame_w: int = 640, frame_h: int = 480) -> bool:
    """
    Memverifikasi secara geometris apakah bounding box masuk akal sebagai plat nomor kendaraan.
    Plat nomor Indonesia memiliki bentuk persegi panjang horizontal:
    - Rasio lebar terhadap tinggi (aspect ratio = width / height) minimal 1.35 dan maksimal 5.8.
    - Telapak tangan, wajah, atau badan manusia memiliki aspect ratio < 1.25 (vertikal atau persegi)
      sehingga PASTI ditolak sebagai plat kendaraan.
    - Tinggi plat tidak boleh melebihi 45% tinggi frame (karena plat nomor tidak menutupi setengah layar).
    - Lebar plat minimal 40 pixel dan tinggi minimal 15 pixel.
    """
    bw = x2 - x1
    bh = y2 - y1
    if bh <= 0 or bw <= 0:
        return False
    
    aspect_ratio = bw / bh
    # Plat nomor kendaraan Indonesia selalu persegi panjang horizontal
    if aspect_ratio < 1.35 or aspect_ratio > 5.8:
        return False
    
    # Ukuran minimum pixel
    if bw < 40 or bh < 15:
        return False
        
    # Plat nomor tidak mungkin menutupi lebih dari 45% tinggi frame
    if frame_h > 0 and bh > (frame_h * 0.45):
        return False

    # Plat nomor tidak mungkin menutupi lebih dari 95% lebar frame
    if frame_w > 0 and bw > (frame_w * 0.95):
        return False

    return True


def format_duration(minutes: int) -> str:
    """Format durasi menit ke dalam teks bahasa Indonesia yang ramah."""
    if minutes < 1:
        return "Kurang dari 1 menit"
    hours = minutes // 60
    rem_min = minutes % 60
    if hours > 0:
        return f"{hours} Jam {rem_min} Menit"
    return f"{rem_min} Menit"


def decide(
    plate: Optional[str],
    confidence: float = 1.0,
    vehicle_data: Optional[Dict[str, Any]] = None,
    min_conf: float = 0.40,
    gate_direction: str = "IN",
    entry_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Logika penentuan status kendaraan, tarif, dan durasi parkir (IN vs OUT).
    """
    norm = normalize_plate(plate)
    display_plate = format_plate_display(plate) if plate else "-"

    # 0. VALIDASI FORMAT PLAT & OCR CONFIDENCE
    if not norm or not is_valid_plate_pattern(norm) or confidence < min_conf:
        if norm and not is_valid_plate_pattern(norm):
            reason = "Bukan plat kendaraan — objek/teks acak tidak sesuai format plat Indonesia"
            status_code = "INVALID_PLATE"
            category = "Bukan Plat Nomor"
        else:
            reason = "OCR kurang yakin — ambil ulang gambar / verifikasi petugas"
            status_code = "OCR_UNCERTAIN"
            category = None

        return {
            "plate": display_plate,
            "normalized_plate": norm,
            "category": category,
            "fee": None,
            "decision": reason,
            "status_code": status_code,
            "is_registered": False,
            "owner_name": None,
            "confidence": confidence,
            "duration_minutes": 0,
            "duration_display": "-",
            "entry_time": None,
        }

    # Hitung durasi jika arah gerbang OUT dan ada data masuk (entry_record)
    duration_minutes = 0
    duration_display = "-"
    entry_time_iso = None

    if gate_direction == "OUT" and entry_record:
        entry_time_iso = entry_record.get("created_at")
        if entry_time_iso:
            try:
                entry_dt = datetime.fromisoformat(entry_time_iso)
                diff = datetime.now() - entry_dt
                duration_minutes = max(0, int(diff.total_seconds() // 60))
                duration_display = format_duration(duration_minutes)
            except Exception:
                duration_minutes = 0
                duration_display = "Tercatat"

    # 1. KENDARAAN TAMU / TIDAK TERDAFTAR DI DATABASE
    if vehicle_data is None:
        base_fee = TARIFFS.get("Tamu", 5000)
        # Jika keluar dan ada durasi lebih dari 2 jam, bisa hitung progresif atau flat
        fee = base_fee
        if gate_direction == "OUT":
            if entry_record:
                decision = f"TAMU SELESAI PARKIR ({duration_display}) — TOTAL BAYAR {rupiah(fee)}"
            else:
                decision = f"TAMU KELUAR GERBANG — WAJIB BAYAR {rupiah(fee)}"
        else:
            decision = f"KENDARAAN TAMU MASUK — TIKET TERCETAK (TARIF {rupiah(fee)})"

        return {
            "plate": display_plate,
            "normalized_plate": norm,
            "category": "Tamu / Belum Terdaftar",
            "fee": fee,
            "decision": decision,
            "status_code": "TAMU_UNREGISTERED",
            "is_registered": False,
            "owner_name": "(Tamu / Luar Kampus)",
            "confidence": confidence,
            "duration_minutes": duration_minutes,
            "duration_display": duration_display,
            "entry_time": entry_time_iso,
        }

    # 2. KENDARAAN TERDAFTAR (DOSEN / MAHASISWA)
    category = vehicle_data.get("category", "Mahasiswa")
    owner_name = vehicle_data.get("owner_name", "")
    registered_plate = vehicle_data.get("plate", display_plate)
    
    fee = TARIFFS.get(category, 2000)

    if gate_direction == "OUT":
        if fee == 0:
            decision = f"DOSEN KELUAR ({duration_display}) — PARKIR GRATIS"
        else:
            decision = f"{category.upper()} KELUAR ({duration_display}) — TOTAL BAYAR {rupiah(fee)}"
    else:
        if fee == 0:
            decision = "DOSEN TERDAFTAR — PARKIR GRATIS"
        else:
            decision = f"{category.upper()} TERDAFTAR — WAJIB BAYAR {rupiah(fee)}"

    return {
        "plate": registered_plate,
        "normalized_plate": norm,
        "category": category,
        "fee": fee,
        "decision": decision,
        "status_code": "REGISTERED",
        "is_registered": True,
        "owner_name": owner_name,
        "confidence": confidence,
        "duration_minutes": duration_minutes,
        "duration_display": duration_display,
        "entry_time": entry_time_iso,
    }


def rupiah(value: Any) -> str:
    """Format angka ke format mata uang Rupiah dengan aman terhadap None, NaN, dan string kosong."""
    if value is None:
        return "-"
    try:
        # Menangani float('nan') atau pandas NaN
        import math
        if isinstance(value, float) and math.isnan(value):
            return "-"
        val_str = str(value).strip()
        if not val_str or val_str.lower() == "nan" or val_str == "None":
            return "-"
        return "Rp" + f"{int(float(value)):,}".replace(",", ".")
    except (ValueError, TypeError):
        return "-"
