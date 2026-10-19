import os
import tempfile
import unittest
from pathlib import Path

from parking_logic import (
    normalize_plate,
    format_plate_display,
    disambiguate_plate,
    is_valid_plate_pattern,
    format_duration,
    decide,
    rupiah,
    TARIFFS,
    DEFAULT_TARIFFS,
)
import db_manager
from dataset_validator import (
    validate_yolo_annotation_line,
    validate_dataset,
)


class TestParkingLogic(unittest.TestCase):
    def test_normalize_plate(self):
        self.assertEqual(normalize_plate("b 1234 xyz"), "B1234XYZ")
        self.assertEqual(normalize_plate(" B-5678 ABC "), "B5678ABC")
        self.assertEqual(normalize_plate("DK.9999.ZZ"), "DK9999ZZ")
        self.assertEqual(normalize_plate(""), "")
        self.assertEqual(normalize_plate(None), "")

    def test_format_plate_display(self):
        self.assertEqual(format_plate_display("b  1234  xyz"), "B 1234 XYZ")
        self.assertEqual(format_plate_display("B-1234-XYZ"), "B 1234 XYZ")

    def test_disambiguate_plate(self):
        # O in middle number converted to 0, I converted to 1
        self.assertEqual(disambiguate_plate("B 12O4 XYZ"), "B1204XYZ")
        self.assertEqual(disambiguate_plate("B 1I34 XYZ"), "B1134XYZ")

    def test_tariffs(self):
        self.assertEqual(TARIFFS["Dosen"], 0)
        self.assertEqual(TARIFFS["Mahasiswa"], 2000)
        self.assertEqual(TARIFFS["Tamu"], 5000)

    def test_decide_registered(self):
        veh = {"plate": "B 1234 XYZ", "owner_name": "Dr. Andi", "category": "Dosen"}
        res = decide("B 1234 XYZ", confidence=0.9, vehicle_data=veh)
        self.assertTrue(res["is_registered"])
        self.assertEqual(res["fee"], 0)
        self.assertIn("GRATIS", res["decision"].upper())
        self.assertEqual(res["status_code"], "REGISTERED")

        veh_mhs = {"plate": "B 5678 ABC", "owner_name": "Budi", "category": "Mahasiswa"}
        res_mhs = decide("B 5678 ABC", confidence=0.85, vehicle_data=veh_mhs)
        self.assertTrue(res_mhs["is_registered"])
        self.assertEqual(res_mhs["fee"], 2000)
        self.assertIn("2.000", res_mhs["decision"])

    def test_decide_unregistered(self):
        res = decide("B 9999 UNK", confidence=0.85, vehicle_data=None)
        self.assertFalse(res["is_registered"])
        self.assertEqual(res["fee"], 5000)
        self.assertEqual(res["status_code"], "TAMU_UNREGISTERED")
        self.assertIn("TAMU", res["decision"])

    def test_decide_uncertain_ocr(self):
        veh = {"plate": "B 1234 XYZ", "owner_name": "Dr. Andi", "category": "Dosen"}
        res = decide("B 1234 XYZ", confidence=0.35, vehicle_data=veh, min_conf=0.40)
        self.assertEqual(res["status_code"], "OCR_UNCERTAIN")
        self.assertIn("OCR kurang yakin", res["decision"])

    def test_is_valid_plate_pattern(self):
        # Format plat nomor Indonesia valid
        self.assertTrue(is_valid_plate_pattern("B 1234 XYZ"))
        self.assertTrue(is_valid_plate_pattern("F 4526 FCF"))
        self.assertTrue(is_valid_plate_pattern("B 6802 JPN"))
        self.assertTrue(is_valid_plate_pattern("D 1 A"))
        self.assertTrue(is_valid_plate_pattern("DK 9999 ZZ"))
        self.assertTrue(is_valid_plate_pattern("B 123 QWE"))

        # Benda acak, wajah, tulisan baju, angka/huruf tanpa kombinasi yang sah
        self.assertFalse(is_valid_plate_pattern("NIKE"))
        self.assertFalse(is_valid_plate_pattern("WAJAH"))
        self.assertFalse(is_valid_plate_pattern("12345"))
        self.assertFalse(is_valid_plate_pattern("ABC"))
        self.assertFalse(is_valid_plate_pattern("MUKA SAYA"))
        self.assertFalse(is_valid_plate_pattern(""))
        self.assertFalse(is_valid_plate_pattern(None))

    def test_decide_invalid_plate_format(self):
        # Scan wajah/benda acak tidak boleh dianggap Tamu, harus INVALID_PLATE
        res = decide("WAJAH", confidence=0.9, vehicle_data=None)
        self.assertEqual(res["status_code"], "INVALID_PLATE")
        self.assertIsNone(res["fee"])
        self.assertFalse(res["is_registered"])
        self.assertEqual(res["category"], "Bukan Plat Nomor")
        self.assertIn("Bukan plat kendaraan", res["decision"])

    def test_format_duration(self):
        self.assertEqual(format_duration(0), "Kurang dari 1 menit")
        self.assertEqual(format_duration(45), "45 Menit")
        self.assertEqual(format_duration(90), "1 Jam 30 Menit")
        self.assertEqual(format_duration(120), "2 Jam 0 Menit")

    def test_decide_out_with_entry(self):
        from datetime import datetime, timedelta
        entry_time = (datetime.now() - timedelta(minutes=45)).isoformat(timespec="seconds")
        entry_rec = {"created_at": entry_time, "gate_direction": "IN"}
        
        # Dosen OUT -> free, duration computed
        veh_dosen = {"plate": "B 1234 XYZ", "owner_name": "Dr. Andi", "category": "Dosen"}
        res_d = decide("B 1234 XYZ", confidence=0.9, vehicle_data=veh_dosen, gate_direction="OUT", entry_record=entry_rec)
        self.assertEqual(res_d["fee"], 0)
        self.assertIn("45 Menit", res_d["duration_display"])
        self.assertIn("GRATIS", res_d["decision"])

        # Tamu OUT -> Rp5.000, duration computed
        res_t = decide("B 9999 UNK", confidence=0.85, vehicle_data=None, gate_direction="OUT", entry_record=entry_rec)
        self.assertEqual(res_t["fee"], 5000)
        self.assertIn("45 Menit", res_t["duration_display"])
        self.assertIn("5.000", res_t["decision"])


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        db_manager.init_db(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_upsert_and_find_vehicle(self):
        is_update, msg = db_manager.upsert_vehicle("B 1234 XYZ", "Prof. Hartono", "Dosen", db_path=self.db_path)
        self.assertFalse(is_update)

        # Cari dengan spasi berbeda dan huruf kecil
        found = db_manager.find_vehicle("b1234xyz", db_path=self.db_path)
        self.assertIsNotNone(found)
        self.assertEqual(found["owner_name"], "Prof. Hartono")
        self.assertEqual(found["category"], "Dosen")

        # Update data yang sama
        is_update, msg = db_manager.upsert_vehicle("B 1234 XYZ", "Prof. Hartono M.Sc", "Dosen", db_path=self.db_path)
        self.assertTrue(is_update)
        updated = db_manager.find_vehicle("B 1234 XYZ", db_path=self.db_path)
        self.assertEqual(updated["owner_name"], "Prof. Hartono M.Sc")

    def test_duplicate_prevention(self):
        db_manager.save_transaction(
            plate="B 8888 AAA",
            owner_name="Test Owner",
            category="Tamu",
            fee=5000,
            decision="Wajib bayar",
            status="CONFIRMED",
            gate_direction="IN",
            db_path=self.db_path
        )
        recent = db_manager.check_recent_transaction("B 8888 AAA", cooldown_seconds=60, gate_direction="IN", db_path=self.db_path)
        self.assertIsNotNone(recent)
        self.assertEqual(recent["plate"], "B 8888 AAA")

        # Cek arah gerbang berbeda (OUT) harus None
        recent_out = db_manager.check_recent_transaction("B 8888 AAA", cooldown_seconds=60, gate_direction="OUT", db_path=self.db_path)
        self.assertIsNone(recent_out)

    def test_dashboard_metrics(self):
        db_manager.upsert_vehicle("B 1111 DOS", "Dosen 1", "Dosen", db_path=self.db_path)
        db_manager.upsert_vehicle("B 2222 MHS", "Mhs 1", "Mahasiswa", db_path=self.db_path)

        db_manager.save_transaction("B 1111 DOS", "Dosen 1", "Dosen", 0, "Gratis", db_path=self.db_path)
        db_manager.save_transaction("B 2222 MHS", "Mhs 1", "Mahasiswa", 2000, "Wajib bayar", db_path=self.db_path)
        db_manager.save_transaction("D 9999 UNK", None, "Belum terdaftar", None, "Verifikasi", db_path=self.db_path)

        metrics = db_manager.get_dashboard_metrics(db_path=self.db_path)
        self.assertEqual(metrics["total_vehicles_registered"], 2)
        self.assertEqual(metrics["total_examinations"], 3)
        self.assertEqual(metrics["total_registered_inspected"], 2)
        self.assertEqual(metrics["total_unregistered_inspected"], 1)
        self.assertEqual(metrics["total_revenue"], 2000)

    def test_active_entry_and_currently_parked(self):
        # Awal: kosong
        self.assertEqual(db_manager.get_currently_parked_count(db_path=self.db_path), 0)
        self.assertIsNone(db_manager.get_active_entry("B 1234 XYZ", db_path=self.db_path))

        # Mobil 1 masuk
        db_manager.save_transaction("B 1234 XYZ", "Budi", "Mahasiswa", 2000, "Masuk", gate_direction="IN", db_path=self.db_path)
        self.assertEqual(db_manager.get_currently_parked_count(db_path=self.db_path), 1)
        active = db_manager.get_active_entry("b 1234 xyz", db_path=self.db_path)
        self.assertIsNotNone(active)
        self.assertEqual(active["gate_direction"], "IN")

        # Mobil 2 masuk
        db_manager.save_transaction("B 5555 TAM", "Tamu 1", "Tamu", 5000, "Masuk", gate_direction="IN", db_path=self.db_path)
        self.assertEqual(db_manager.get_currently_parked_count(db_path=self.db_path), 2)

        # Mobil 1 keluar
        db_manager.save_transaction("B 1234 XYZ", "Budi", "Mahasiswa", 2000, "Keluar", gate_direction="OUT", duration_minutes=30, db_path=self.db_path)
        self.assertEqual(db_manager.get_currently_parked_count(db_path=self.db_path), 1)
        self.assertIsNone(db_manager.get_active_entry("b 1234 xyz", db_path=self.db_path))

        # Mobil 2 keluar
        db_manager.save_transaction("B 5555 TAM", "Tamu 1", "Tamu", 5000, "Keluar", gate_direction="OUT", duration_minutes=45, db_path=self.db_path)
        self.assertEqual(db_manager.get_currently_parked_count(db_path=self.db_path), 0)


class TestDatasetValidator(unittest.TestCase):
    def test_validate_annotation_line(self):
        # Valid line: 0 x y w h
        valid, err, cid, coords = validate_yolo_annotation_line("0 0.5 0.5 0.2 0.1")
        self.assertTrue(valid)
        self.assertEqual(cid, 0)
        self.assertEqual(coords, [0.5, 0.5, 0.2, 0.1])

        # Invalid line: out of bounds
        valid, err, cid, coords = validate_yolo_annotation_line("0 1.5 0.5 0.2 0.1")
        self.assertFalse(valid)
        self.assertIn("luar batas", err)

        # Invalid line: non-integer class
        valid, err, cid, coords = validate_yolo_annotation_line("plate 0.5 0.5 0.2 0.1")
        self.assertFalse(valid)
        self.assertIn("bukan integer", err)

        # Invalid line: non-float coordinate
        valid, err, cid, coords = validate_yolo_annotation_line("0 abc 0.5 0.2 0.1")
        self.assertFalse(valid)

    def test_validate_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rep = validate_dataset(tmpdir)
            self.assertFalse(rep["is_valid"])
            self.assertEqual(rep["train_images_count"], 0)


if __name__ == "__main__":
    unittest.main()
