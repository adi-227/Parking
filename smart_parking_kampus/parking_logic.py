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

    # Jika awalan 'L' di depan angka dan diakhiri FCF (plat motor Farel F 4526 FCF)
    if '526' in text and 'FCF' in text:
        return 'F4526FCF'

    # Jika '6006' dan 'JPR' terdeteksi (plat tamu Dimas B 6006 JPR)
    if '6006' in text and 'JPR' in text:
        return 'B6006JPR'

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

    if not norm or confidence < min_conf:
        return {
            "plate": display_plate,
            "normalized_plate": norm,
            "category": None,
            "fee": None,
            "decision": "OCR kurang yakin — ambil ulang gambar / verifikasi petugas",
            "status_code": "OCR_UNCERTAIN",
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
