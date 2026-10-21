import os
import re
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from parking_logic import (
    normalize_plate,
    format_plate_display,
    CATEGORIES,
    TARIFFS,
    format_duration,
    calculate_parking_fee,
    rupiah,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(APP_DIR, "parking.db")


import contextlib

@contextlib.contextmanager
def get_connection(db_path: str = DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Inisialisasi tabel SQLite dengan migrasi kolom jika diperlukan tanpa menghapus data lama."""
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicles (
                plate TEXT PRIMARY KEY,
                owner_name TEXT NOT NULL,
                category TEXT NOT NULL CHECK(category IN ('Dosen','Mahasiswa','Tamu')),
                created_at TEXT
            )
        """)
        # Cek apakah kolom created_at ada pada vehicles (migrasi aman)
        cursor = conn.execute("PRAGMA table_info(vehicles)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "created_at" not in columns:
            conn.execute("ALTER TABLE vehicles ADD COLUMN created_at TEXT")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL,
                owner_name TEXT,
                category TEXT,
                fee INTEGER,
                decision TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'CONFIRMED',
                gate_direction TEXT NOT NULL DEFAULT 'IN',
                notes TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Migrasi kolom tambahan pada transactions jika belum ada
        cursor = conn.execute("PRAGMA table_info(transactions)")
        tx_cols = [row["name"] for row in cursor.fetchall()]
        if "owner_name" not in tx_cols:
            conn.execute("ALTER TABLE transactions ADD COLUMN owner_name TEXT")
        if "status" not in tx_cols:
            conn.execute("ALTER TABLE transactions ADD COLUMN status TEXT NOT NULL DEFAULT 'CONFIRMED'")
        if "gate_direction" not in tx_cols:
            conn.execute("ALTER TABLE transactions ADD COLUMN gate_direction TEXT NOT NULL DEFAULT 'IN'")
        if "notes" not in tx_cols:
            conn.execute("ALTER TABLE transactions ADD COLUMN notes TEXT")
        if "duration_minutes" not in tx_cols:
            conn.execute("ALTER TABLE transactions ADD COLUMN duration_minutes INTEGER DEFAULT 0")
        if "entry_time" not in tx_cols:
            conn.execute("ALTER TABLE transactions ADD COLUMN entry_time TEXT")


def upsert_vehicle(plate: str, owner_name: str, category: str, db_path: str = DEFAULT_DB_PATH) -> Tuple[bool, str]:
    """
    Tambah atau perbarui kendaraan.
    Mengembalikan (is_update, pesan).
    """
    clean_plate = format_plate_display(plate)
    norm = normalize_plate(plate)
    owner = (owner_name or "").strip()
    
    if not norm or len(norm) < 3:
        raise ValueError("Nomor plat tidak valid.")
    if not owner:
        raise ValueError("Nama pemilik wajib diisi.")
    if category not in CATEGORIES:
        raise ValueError(f"Kategori tidak valid. Pilih dari: {', '.join(CATEGORIES)}")

    existing = find_vehicle(clean_plate, db_path=db_path)
    is_update = existing is not None
    now_iso = datetime.now().isoformat(timespec="seconds")

    with get_connection(db_path) as conn:
        if is_update:
            conn.execute("""
                UPDATE vehicles 
                SET owner_name = ?, category = ?
                WHERE plate = ?
            """, (owner, category, existing["plate"]))
            msg = f"Data kendaraan {existing['plate']} berhasil diperbarui."
        else:
            conn.execute("""
                INSERT INTO vehicles (plate, owner_name, category, created_at)
                VALUES (?, ?, ?, ?)
            """, (clean_plate, owner, category, now_iso))
            msg = f"Kendaraan baru {clean_plate} berhasil didaftarkan."
            
    return is_update, msg


def update_vehicle(
    old_plate: str,
    new_owner_name: str,
    new_category: str,
    new_plate: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """
    Perbarui informasi kendaraan (nama pemilik, kategori, dan opsional plat baru).
    Mengembalikan True jika berhasil diperbarui.
    """
    norm_old = normalize_plate(old_plate)
    if not norm_old:
        return False
    target_clean = format_plate_display(new_plate) if new_plate else format_plate_display(old_plate)
    owner = (new_owner_name or "").strip()
    if not owner or new_category not in CATEGORIES:
        return False

    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT plate FROM vehicles").fetchall()
        matched_plate = None
        for r in rows:
            if normalize_plate(r["plate"]) == norm_old:
                matched_plate = r["plate"]
                break
        if not matched_plate:
            return False

        conn.execute("""
            UPDATE vehicles
            SET plate = ?, owner_name = ?, category = ?
            WHERE plate = ?
        """, (target_clean, owner, new_category, matched_plate))

        if target_clean != matched_plate:
            all_tx = conn.execute("SELECT id, plate FROM transactions").fetchall()
            tx_ids_to_update = [t["id"] for t in all_tx if normalize_plate(t["plate"]) == norm_old]
            for tid in tx_ids_to_update:
                conn.execute("UPDATE transactions SET plate = ? WHERE id = ?", (target_clean, tid))

        return True


def delete_vehicle(plate: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Hapus kendaraan berdasarkan plat."""
    with get_connection(db_path) as conn:
        res = conn.execute("DELETE FROM vehicles WHERE plate = ?", (plate,))
        return res.rowcount > 0


def find_vehicle(plate: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Pencarian kendaraan berbasis nomor plat yang dinormalisasi."""
    key = normalize_plate(plate)
    if not key:
        return None
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM vehicles").fetchall()
    for row in rows:
        if normalize_plate(row["plate"]) == key:
            return dict(row)
    return None


def list_vehicles(query: Optional[str] = None, category_filter: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Ambil daftar kendaraan dengan pencarian, filter, serta status parkir terkini (Di Dalam vs Di Luar)."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT plate, owner_name, category, created_at FROM vehicles ORDER BY category, plate").fetchall()
        tx_rows = conn.execute("SELECT plate, gate_direction, created_at FROM transactions ORDER BY id ASC").fetchall()
    
    # Hitung status parkir aktif per plat
    parked_map = {}
    for tx in tx_rows:
        n = normalize_plate(tx["plate"])
        if n:
            parked_map[n] = (tx["gate_direction"], tx["created_at"])
    
    results = []
    norm_query = normalize_plate(query) if query else ""
    raw_query = (query or "").lower().strip()
    now = datetime.now()

    for r in rows:
        item = dict(r)
        if category_filter and category_filter != "Semua" and item["category"] != category_filter:
            continue
        if query:
            match_plate = norm_query in normalize_plate(item["plate"])
            match_owner = raw_query in (item["owner_name"] or "").lower()
            if not (match_plate or match_owner):
                continue
        
        # Tambahkan status parkir real-time
        n_plate = normalize_plate(item["plate"])
        last_dir, last_time = parked_map.get(n_plate, (None, None))
        if last_dir == "IN":
            dur_txt = ""
            if last_time:
                try:
                    diff = now - datetime.fromisoformat(last_time)
                    mins = max(0, int(diff.total_seconds() // 60))
                    dur_txt = f" ({mins}m)"
                except Exception:
                    pass
            item["is_parked"] = True
            item["parking_status"] = f"🟢 Di Dalam{dur_txt}"
            item["entry_time"] = last_time
        else:
            item["is_parked"] = False
            item["parking_status"] = "⚪ Di Luar"
            item["entry_time"] = None

        results.append(item)
    return results


def check_recent_transaction(plate: str, cooldown_seconds: int = 60, gate_direction: str = "IN", db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Pencegahan transaksi ganda:
    Periksa apakah kendaraan yang sama sudah tercatat dalam rentang cooldown.
    """
    norm = normalize_plate(plate)
    if not norm:
        return None
    
    cutoff = (datetime.now() - timedelta(seconds=cooldown_seconds)).isoformat(timespec="seconds")
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM transactions 
               WHERE gate_direction = ? AND created_at >= ? 
               ORDER BY id DESC""",
            (gate_direction, cutoff)
        ).fetchall()
        
    for r in rows:
        if normalize_plate(r["plate"]) == norm:
            return dict(r)
    return None


def get_active_entry(plate: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Mencari transaksi masuk (IN) terakhir untuk plat ini yang belum ditutup oleh transaksi OUT."""
    norm = normalize_plate(plate)
    if not norm:
        return None
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM transactions 
               ORDER BY id DESC"""
        ).fetchall()
    
    # Periksa transaksi paling baru untuk plat ini
    for r in rows:
        if normalize_plate(r["plate"]) == norm:
            # Jika transaksi terakhir adalah 'IN', berarti masih aktif di dalam
            if r["gate_direction"] == "IN":
                return dict(r)
            else:
                # Transaksi terakhir sudah 'OUT', berarti sudah keluar
                return None
    return None


def get_currently_parked_count(db_path: str = DEFAULT_DB_PATH) -> int:
    """Menghitung jumlah kendaraan yang saat ini masih berada di dalam area parkir kampus."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT plate, gate_direction FROM transactions ORDER BY id ASC").fetchall()
    
    status_map = {}
    for r in rows:
        norm = normalize_plate(r["plate"])
        if norm:
            status_map[norm] = r["gate_direction"]
    
    return sum(1 for direction in status_map.values() if direction == "IN")


def save_transaction(
    plate: str,
    owner_name: Optional[str],
    category: Optional[str],
    fee: Optional[int],
    decision: str,
    status: str = "CONFIRMED",
    gate_direction: str = "IN",
    notes: Optional[str] = None,
    duration_minutes: Optional[int] = 0,
    entry_time: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Menyimpan transaksi yang telah dikonfirmasi oleh petugas."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """INSERT INTO transactions(plate, owner_name, category, fee, decision, status, gate_direction, notes, duration_minutes, entry_time, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                format_plate_display(plate) if plate else "-",
                owner_name,
                category,
                fee,
                decision,
                status,
                gate_direction,
                notes,
                duration_minutes or 0,
                entry_time,
                datetime.now().isoformat(timespec="seconds")
            ),
        )
        return cursor.lastrowid


def list_transactions(
    limit: int = 300,
    plate_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """Ambil daftar transaksi dengan filter dan batas jumlah."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT id, plate, owner_name, category, fee, decision, status, gate_direction, notes, duration_minutes, entry_time, created_at 
               FROM transactions ORDER BY id DESC LIMIT ?""",
            (limit,)
        ).fetchall()

    norm_query = normalize_plate(plate_filter) if plate_filter else ""
    results = []
    for r in rows:
        item = dict(r)
        if category_filter and category_filter != "Semua" and item["category"] != category_filter:
            continue
        if plate_filter and norm_query not in normalize_plate(item["plate"]):
            continue
        if start_date and item["created_at"][:10] < start_date:
            continue
        if end_date and item["created_at"][:10] > end_date:
            continue
        results.append(item)
    return results


def get_dashboard_metrics(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Menghitung metrik agregat untuk dashboard ringkasan."""
    with get_connection(db_path) as conn:
        total_registered = conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
        total_tx = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        total_fee = conn.execute("SELECT SUM(fee) FROM transactions WHERE fee IS NOT NULL AND status = 'CONFIRMED'").fetchone()[0] or 0
        total_registered_tx = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE category IN ('Dosen', 'Mahasiswa')"
        ).fetchone()[0]
        total_unregistered_tx = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE category IS NULL OR category LIKE '%Tamu%' OR category = 'Belum terdaftar'"
        ).fetchone()[0]

    currently_parked = get_currently_parked_count(db_path)

    return {
        "total_vehicles_registered": total_registered,
        "total_examinations": total_tx,
        "total_registered_inspected": total_registered_tx,
        "total_unregistered_inspected": total_unregistered_tx,
        "total_revenue": total_fee,
        "currently_parked": currently_parked,
    }


def get_currently_parked_vehicles(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Mengambil daftar seluruh kendaraan (terdaftar maupun tamu) yang saat ini
    masih berstatus 'IN' (sedang parkir di dalam area kampus).
    """
    with get_connection(db_path) as conn:
        tx_rows = conn.execute("""
            SELECT id, plate, owner_name, category, fee, gate_direction, notes, created_at 
            FROM transactions 
            ORDER BY id ASC
        """).fetchall()

    last_tx_map = {}
    for r in tx_rows:
        norm = normalize_plate(r["plate"])
        if norm:
            last_tx_map[norm] = dict(r)

    now = datetime.now()
    active_list = []
    for norm, tx in last_tx_map.items():
        if tx.get("gate_direction") == "IN":
            entry_iso = tx.get("created_at")
            dur_mins = 0
            dur_text = "Kurang dari 1 menit"
            if entry_iso:
                try:
                    dt = datetime.fromisoformat(entry_iso)
                    diff = now - dt
                    dur_mins = max(0, int(diff.total_seconds() // 60))
                    dur_text = format_duration(dur_mins)
                except Exception:
                    pass
            tx["duration_minutes"] = dur_mins
            tx["duration_text"] = dur_text
            cat = tx.get("category") or "Tamu"
            tx["current_fee"] = calculate_parking_fee(cat, dur_mins)
            active_list.append(tx)

    active_list.sort(key=lambda x: x.get("id", 0), reverse=True)
    return active_list


def delete_transaction(tx_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Menghapus data satu transaksi berdasarkan ID."""
    with get_connection(db_path) as conn:
        res = conn.execute("DELETE FROM transactions WHERE id = ?", (tx_id,))
        return res.rowcount > 0


def delete_completed_transactions(db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Menghapus riwayat transaksi yang sesinya sudah selesai (kendaraan sudah OUT),
    sehingga riwayat tidak menumpuk, tanpa menghapus sesi kendaraan yang sedang parkir (IN).
    """
    with get_connection(db_path) as conn:
        active_vehicles = get_currently_parked_vehicles(db_path)
        active_ids = {v["id"] for v in active_vehicles if "id" in v}

        all_tx = conn.execute("SELECT id FROM transactions").fetchall()
        ids_to_delete = [t["id"] for t in all_tx if t["id"] not in active_ids]

        if ids_to_delete:
            chunk_size = 500
            for i in range(0, len(ids_to_delete), chunk_size):
                chunk = ids_to_delete[i:i + chunk_size]
                placeholders = ",".join(["?"] * len(chunk))
                conn.execute(f"DELETE FROM transactions WHERE id IN ({placeholders})", chunk)
        return len(ids_to_delete)


def clear_all_transactions(db_path: str = DEFAULT_DB_PATH) -> int:
    """Menghapus seluruh riwayat transaksi (reset data transaksi)."""
    with get_connection(db_path) as conn:
        res = conn.execute("DELETE FROM transactions")
        return res.rowcount


def update_transaction(
    tx_id: int,
    plate: Optional[str] = None,
    owner_name: Optional[str] = None,
    category: Optional[str] = None,
    fee: Optional[int] = None,
    gate_direction: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """Memperbarui informasi, nomor plat, kategori, tarif, arah gerbang, atau catatan pada transaksi tertentu."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM transactions WHERE id = ?", (tx_id,)).fetchone()
        if not row:
            return False
        new_plate = format_plate_display(plate) if plate is not None else row["plate"]
        new_owner = owner_name if owner_name is not None else row["owner_name"]
        new_cat = category if category is not None else row["category"]
        new_fee = fee if fee is not None else row["fee"]
        new_gate = gate_direction if gate_direction is not None else row["gate_direction"]
        new_notes = notes if notes is not None else row["notes"]
        conn.execute("""
            UPDATE transactions
            SET plate = ?, owner_name = ?, category = ?, fee = ?, gate_direction = ?, notes = ?
            WHERE id = ?
        """, (new_plate, new_owner, new_cat, new_fee, new_gate, new_notes, tx_id))
        return True


def checkout_vehicle(
    plate: str,
    fee: Optional[int] = None,
    notes: Optional[str] = "Checkout Manual Petugas",
    db_path: str = DEFAULT_DB_PATH
) -> Optional[int]:
    """
    Menyelesaikan sesi parkir kendaraan (mencatat transaksi OUT).
    Jika fee None, otomatis dihitung berdasarkan kategori dan durasi parkir.
    """
    norm = normalize_plate(plate)
    if not norm:
        return None

    active = get_active_entry(plate, db_path=db_path)
    if not active:
        return None

    entry_iso = active.get("created_at")
    dur_mins = 0
    if entry_iso:
        try:
            diff = datetime.now() - datetime.fromisoformat(entry_iso)
            dur_mins = max(0, int(diff.total_seconds() // 60))
        except Exception:
            pass

    cat = active.get("category") or "Tamu"
    calc_fee = calculate_parking_fee(cat, dur_mins) if fee is None else fee

    out_id = save_transaction(
        plate=format_plate_display(plate),
        owner_name=active.get("owner_name"),
        category=cat,
        fee=calc_fee,
        decision=f"CHECKOUT KELUAR ({format_duration(dur_mins)}) — {rupiah(calc_fee)}",
        status="CONFIRMED",
        gate_direction="OUT",
        notes=notes or "Checkout Manual",
        duration_minutes=dur_mins,
        entry_time=entry_iso,
        db_path=db_path
    )
    return out_id


def delete_active_parked_sessions(
    tx_id: Optional[int] = None,
    plate: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Menghapus transaksi sesi parkir yang sedang aktif (misal sisa data testing kemarin).
    Jika tx_id diberikan: menghapus transaksi ID tersebut.
    Jika plate diberikan: menghapus sesi aktif plat tersebut.
    Jika keduanya None: menghapus SEMUA sesi kendaraan yang saat ini sedang parkir (IN),
    sehingga jumlah kendaraan terparkir seketika menjadi 0.
    """
    active_vehicles = get_currently_parked_vehicles(db_path)
    if not active_vehicles:
        return 0

    with get_connection(db_path) as conn:
        if tx_id is not None:
            res = conn.execute("DELETE FROM transactions WHERE id = ?", (tx_id,))
            return res.rowcount
        elif plate is not None:
            norm = normalize_plate(plate)
            target_ids = [v["id"] for v in active_vehicles if normalize_plate(v["plate"]) == norm]
            if target_ids:
                ph = ",".join(["?"] * len(target_ids))
                res = conn.execute(f"DELETE FROM transactions WHERE id IN ({ph})", target_ids)
                return res.rowcount
            return 0
        else:
            all_active_ids = [v["id"] for v in active_vehicles]
            if all_active_ids:
                ph = ",".join(["?"] * len(all_active_ids))
                res = conn.execute(f"DELETE FROM transactions WHERE id IN ({ph})", all_active_ids)
                return res.rowcount
            return 0


def checkout_all_active_vehicles(
    notes: str = "Checkout Massal / Penutupan Sesi",
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Mencatat transaksi OUT untuk seluruh kendaraan yang saat ini sedang parkir,
    sehingga seluruh sesi parkir terselesaikan secara resmi.
    """
    active_vehicles = get_currently_parked_vehicles(db_path)
    count = 0
    for v in active_vehicles:
        res = checkout_vehicle(v["plate"], notes=notes, db_path=db_path)
        if res:
            count += 1
    return count

