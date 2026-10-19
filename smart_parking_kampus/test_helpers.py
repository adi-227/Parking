import os
import tempfile
import unittest
from pathlib import Path

from parking_logic import (
    normalize_plate,
    format_plate_display,
    disambiguate_plate,
    is_valid_plate_pattern,
    is_plausible_plate_box,
    extract_candidate_plate_boxes,
    compute_iou,
    SmoothBoxTracker,
    format_duration,
    decide,
    rupiah,
    TARIFFS,
    DEFAULT_TARIFFS,
    calculate_parking_fee,
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
        self.assertEqual(format_plate_display("F4220FGM"), "F 4220 FGM")
        self.assertEqual(format_plate_display("B6802JPN"), "B 6802 JPN")
        self.assertEqual(format_plate_display("F4526FCF"), "F 4526 FCF")

    def test_disambiguate_plate(self):
        # O in middle number converted to 0, I converted to 1
        self.assertEqual(disambiguate_plate("B 12O4 XYZ"), "B1204XYZ")
        self.assertEqual(disambiguate_plate("B 1I34 XYZ"), "B1134XYZ")
        # Leading / trailing border removal and B 6991 JGU
        self.assertEqual(disambiguate_plate("IB 6991 JGU"), "B6991JGU")
        self.assertEqual(disambiguate_plate("IB 6991 JCUL"), "B6991JGU")
        self.assertEqual(disambiguate_plate("B 6991 JGU"), "B6991JGU")
        # Motorcycle plate F 4220 FGM and its common OCR misreadings
        self.assertEqual(disambiguate_plate("F 4220 FGM"), "F4220FGM")
        self.assertEqual(disambiguate_plate("FL220 FGH"), "F4220FGM")
        self.assertEqual(disambiguate_plate("EL220 FGD"), "F4220FGM")
        self.assertEqual(disambiguate_plate("L220EGH"), "F4220FGM")

    def test_tariffs(self):
        self.assertEqual(TARIFFS["Dosen"], 0)
        self.assertEqual(TARIFFS["Mahasiswa"], 2000)
        self.assertEqual(TARIFFS["Tamu"], 3000)

    def test_calculate_parking_fee(self):
        # Dosen selalu gratis (Rp 0)
        self.assertEqual(calculate_parking_fee("Dosen", 0), 0)
        self.assertEqual(calculate_parking_fee("Dosen", 300), 0)
        # Mahasiswa flat Rp 2.000 seharian
        self.assertEqual(calculate_parking_fee("Mahasiswa", 10), 2000)
        self.assertEqual(calculate_parking_fee("Mahasiswa", 600), 2000)
        # Tamu: 1 jam pertama Rp 3.000
        self.assertEqual(calculate_parking_fee("Tamu", 0), 3000)
        self.assertEqual(calculate_parking_fee("Tamu", 45), 3000)
        self.assertEqual(calculate_parking_fee("Tamu", 60), 3000)
        # Tamu: Jam ke-2 (61 - 120 menit) -> Rp 3.000 + Rp 2.000 = Rp 5.000
        self.assertEqual(calculate_parking_fee("Tamu", 61), 5000)
        self.assertEqual(calculate_parking_fee("Tamu", 120), 5000)
        # Tamu: Jam ke-3 (121 - 180 menit) -> Rp 3.000 + 2*Rp 2.000 = Rp 7.000
        self.assertEqual(calculate_parking_fee("Tamu", 121), 7000)
        self.assertEqual(calculate_parking_fee("Tamu", 180), 7000)
        # Tamu: Jam ke-4 -> Rp 9.000
        self.assertEqual(calculate_parking_fee("Tamu", 181), 9000)

    def test_decide_unregistered(self):
        # Saat masuk (IN)
        res = decide("B 9999 UNK", confidence=0.85, vehicle_data=None, gate_direction="IN")
        self.assertFalse(res["is_registered"])
        self.assertEqual(res["fee"], 3000)
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

    def test_is_plausible_plate_box(self):
        # 1. Plat motor Indonesia valid (lebar 200px, tinggi 55px -> rasio 3.64)
        self.assertTrue(is_plausible_plate_box(100, 100, 300, 155, 640, 480))
        # 2. Plat mobil Indonesia valid (lebar 300px, tinggi 95px -> rasio 3.16)
        self.assertTrue(is_plausible_plate_box(50, 150, 350, 245, 640, 480))
        # 3. Telapak tangan dari foto user (lebar 185px, tinggi 253px -> rasio 0.73, vertikal)
        self.assertFalse(is_plausible_plate_box(160, 150, 345, 403, 640, 480))
        # 4. Wajah manusia / kepala (lebar 150px, tinggi 160px -> rasio 0.94)
        self.assertFalse(is_plausible_plate_box(200, 100, 350, 260, 640, 480))
        # 5. Kotak terlalu tinggi (menutupi > 50% tinggi layar kamera, misal bh=260 di 480h)
        self.assertFalse(is_plausible_plate_box(50, 50, 500, 310, 640, 480))
        # 6. Kotak noise terlalu kecil
        self.assertFalse(is_plausible_plate_box(10, 10, 25, 20, 640, 480))
        # 7. Plat mobil serong samping kanan / kiri (rasio 1.30)
        self.assertTrue(is_plausible_plate_box(100, 100, 230, 200, 640, 480))
        # 8. Plat motor serong samping tajam ~45 derajat (rasio 1.15)
        self.assertTrue(is_plausible_plate_box(100, 100, 215, 200, 640, 480))

    def test_extract_candidate_plate_boxes(self):
        # 1. Kotak plat mobil standar
        std_boxes = [([50, 150, 350, 245], 0.92)]
        cands = extract_candidate_plate_boxes(std_boxes, frame_w=640, frame_h=480)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0][2], "standard")
        self.assertEqual(cands[0][0], [50, 150, 350, 245])

        # 2. Kotak plat motor menyatu dengan spatbor belakang (w=320, h=345, ar=0.93)
        motor_boxes = [([60, 152, 380, 497], 0.88)]
        cands_m = extract_candidate_plate_boxes(motor_boxes, frame_w=640, frame_h=480)
        self.assertEqual(len(cands_m), 1)
        self.assertEqual(cands_m[0][2], "motorcycle_adjusted")
        # Tinggi kotak disesuaikan ke ~52% bagian atas
        self.assertTrue(cands_m[0][0][3] < 497)
        self.assertTrue(cands_m[0][0][3] > 152)

        # 3. Noise / telapak tangan ditolak
        palm_boxes = [([160, 150, 345, 403], 0.70)]
        cands_p = extract_candidate_plate_boxes(palm_boxes, frame_w=640, frame_h=480)
        # Bounding box telapak tangan setelah disesuaikan tetap bukan proporsi plat kendaraan
        self.assertEqual(len(cands_p), 0)

        # 4. Plat mobil serong samping kanan/kiri lolos deteksi
        angled_boxes = [([100, 100, 230, 200], 0.85)]
        cands_a = extract_candidate_plate_boxes(angled_boxes, frame_w=640, frame_h=480)
        self.assertEqual(len(cands_a), 1)
        self.assertEqual(cands_a[0][2], "standard")

    def test_format_duration(self):
        self.assertEqual(format_duration(0), "Kurang dari 1 menit")
        self.assertEqual(format_duration(45), "45 Menit")
        self.assertEqual(format_duration(90), "1 Jam 30 Menit")
        self.assertEqual(format_duration(120), "2 Jam 0 Menit")

    def test_decide_out_with_entry(self):
        from datetime import datetime, timedelta
        entry_time_45 = (datetime.now() - timedelta(minutes=45)).isoformat(timespec="seconds")
        entry_rec_45 = {"created_at": entry_time_45, "gate_direction": "IN"}

        # Dosen OUT -> free, duration computed
        veh_dosen = {"plate": "B 1234 XYZ", "owner_name": "Dr. Andi", "category": "Dosen"}
        res_d = decide("B 1234 XYZ", confidence=0.9, vehicle_data=veh_dosen, gate_direction="OUT", entry_record=entry_rec_45)
        self.assertEqual(res_d["fee"], 0)
        self.assertIn("45 Menit", res_d["duration_display"])
        self.assertIn("GRATIS", res_d["decision"])

        # Tamu OUT (45 menit <= 1 jam pertama) -> Rp3.000
        res_t1 = decide("B 9999 UNK", confidence=0.85, vehicle_data=None, gate_direction="OUT", entry_record=entry_rec_45)
        self.assertEqual(res_t1["fee"], 3000)
        self.assertIn("45 Menit", res_t1["duration_display"])
        self.assertIn("3.000", res_t1["decision"])

        # Tamu OUT (80 menit = 1 jam 20 menit) -> Rp3.000 + Rp2.000 = Rp5.000
        entry_time_80 = (datetime.now() - timedelta(minutes=80)).isoformat(timespec="seconds")
        entry_rec_80 = {"created_at": entry_time_80, "gate_direction": "IN"}
        res_t2 = decide("B 9999 UNK", confidence=0.85, vehicle_data=None, gate_direction="OUT", entry_record=entry_rec_80)
        self.assertEqual(res_t2["fee"], 5000)
        self.assertIn("1 Jam 20 Menit", res_t2["duration_display"])
        self.assertIn("5.000", res_t2["decision"])

    def test_compute_iou(self):
        # 1. Identical boxes
        boxA = (100, 100, 200, 200)
        self.assertAlmostEqual(compute_iou(boxA, boxA), 1.0)

        # 2. Disjoint boxes
        boxB = (300, 300, 400, 400)
        self.assertEqual(compute_iou(boxA, boxB), 0.0)

        # 3. Partial overlap (50% area overlap)
        # boxA: [0, 0, 10, 10], area = 100
        # boxC: [5, 0, 15, 10], area = 100
        # intersection: [5, 0, 10, 10], area = 50
        # union: 100 + 100 - 50 = 150 -> IoU = 50/150 = 1/3
        box1 = (0, 0, 10, 10)
        box2 = (5, 0, 15, 10)
        self.assertAlmostEqual(compute_iou(box1, box2), 50.0 / 150.0)

    def test_smooth_box_tracker(self):
        tracker = SmoothBoxTracker(alpha=0.6, max_miss_frames=3)

        # 1. First detection: instantaneous response without lag
        box1 = (100, 100, 200, 200)
        res1 = tracker.update([(box1, 0.9, "standard")])
        self.assertEqual(res1, box1)

        # 2. Slight jitter on next frame: coordinates are smoothed (no sudden jerk)
        # Shifted by 10px on x1
        box2 = (110, 100, 210, 200)
        res2 = tracker.update([(box2, 0.9, "standard")])
        # x1 = 0.6 * 110 + 0.4 * 100 = 66 + 40 = 106
        self.assertEqual(res2[0], 106)
        self.assertEqual(res2[1], 100)

        # 3. Persistence / Hysteresis: 1 missed frame does NOT clear the box
        res_miss1 = tracker.update([])
        self.assertIsNotNone(res_miss1)
        self.assertEqual(res_miss1, res2)

        # 4. 2nd missed frame still retains box
        res_miss2 = tracker.update([])
        self.assertIsNotNone(res_miss2)

        # 5. After max_miss_frames exceeded, box cleanly expires
        tracker.update([])
        res_expired = tracker.update([])
        self.assertIsNone(res_expired)
        self.assertIsNone(tracker.get_box())


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

    def test_update_and_delete_vehicle(self):
        # 1. Tambah kendaraan
        db_manager.upsert_vehicle("B 1234 XYZ", "Budi Santoso", "Mahasiswa", db_path=self.db_path)
        # Catat transaksi masuk
        tx_id = db_manager.save_transaction("B 1234 XYZ", "Budi Santoso", "Mahasiswa", 2000, "Masuk", gate_direction="IN", db_path=self.db_path)

        # 2. Update kendaraan (ubah nama dan ganti plat nomor jika ada koreksi)
        up_res = db_manager.update_vehicle(
            old_plate="B 1234 XYZ",
            new_owner_name="Budi Santoso S.T.",
            new_category="Dosen",
            new_plate="B 1234 ABC",
            db_path=self.db_path
        )
        self.assertTrue(up_res)

        # Cari dengan plat baru
        found = db_manager.find_vehicle("B 1234 ABC", db_path=self.db_path)
        self.assertIsNotNone(found)
        self.assertEqual(found["owner_name"], "Budi Santoso S.T.")
        self.assertEqual(found["category"], "Dosen")

        # Pastikan transaksi lama juga terupdate platnya
        txs = db_manager.list_transactions(db_path=self.db_path)
        self.assertEqual(txs[0]["plate"], "B 1234 ABC")

        # 3. Hapus kendaraan
        del_res = db_manager.delete_vehicle("B 1234 ABC", db_path=self.db_path)
        self.assertTrue(del_res)
        self.assertIsNone(db_manager.find_vehicle("B 1234 ABC", db_path=self.db_path))

    def test_list_vehicles_parking_status(self):
        db_manager.upsert_vehicle("B 1000 MHS", "Mhs Satu", "Mahasiswa", db_path=self.db_path)
        db_manager.upsert_vehicle("B 2000 DOS", "Dosen Satu", "Dosen", db_path=self.db_path)

        # Status awal: keduanya di luar
        v_list = db_manager.list_vehicles(db_path=self.db_path)
        status_map = {v["plate"]: v for v in v_list}
        self.assertFalse(status_map["B 1000 MHS"]["is_parked"])
        self.assertEqual(status_map["B 1000 MHS"]["parking_status"], "⚪ Di Luar")
        self.assertFalse(status_map["B 2000 DOS"]["is_parked"])

        # Mhs Satu masuk (IN)
        db_manager.save_transaction("B 1000 MHS", "Mhs Satu", "Mahasiswa", 2000, "Masuk", gate_direction="IN", db_path=self.db_path)
        v_list2 = db_manager.list_vehicles(db_path=self.db_path)
        status_map2 = {v["plate"]: v for v in v_list2}
        self.assertTrue(status_map2["B 1000 MHS"]["is_parked"])
        self.assertIn("🟢 Di Dalam", status_map2["B 1000 MHS"]["parking_status"])
        self.assertFalse(status_map2["B 2000 DOS"]["is_parked"])

    def test_get_currently_parked_vehicles_and_cleanup(self):
        # Kendaraan 1 (Mahasiswa) masuk
        db_manager.save_transaction("B 1111 MHS", "Agus", "Mahasiswa", 2000, "Masuk", gate_direction="IN", db_path=self.db_path)
        # Kendaraan 2 (Tamu) masuk
        db_manager.save_transaction("B 2222 TMU", "Tamu B", "Tamu", 3000, "Masuk", gate_direction="IN", db_path=self.db_path)
        # Kendaraan 3 (Tamu C) masuk lalu keluar
        db_manager.save_transaction("B 3333 OUT", "Tamu C", "Tamu", 3000, "Masuk", gate_direction="IN", db_path=self.db_path)
        db_manager.save_transaction("B 3333 OUT", "Tamu C", "Tamu", 3000, "Keluar", gate_direction="OUT", duration_minutes=30, db_path=self.db_path)

        # Cek kendaraan yang sedang parkir
        active = db_manager.get_currently_parked_vehicles(db_path=self.db_path)
        self.assertEqual(len(active), 2)
        active_plates = [a["plate"] for a in active]
        self.assertIn("B 1111 MHS", active_plates)
        self.assertIn("B 2222 TMU", active_plates)
        self.assertNotIn("B 3333 OUT", active_plates)

        # Test delete single transaction
        tx_to_del = active[0]["id"]
        self.assertTrue(db_manager.delete_transaction(tx_to_del, db_path=self.db_path))

        # Test delete completed transactions (hanya menghapus sesi yang sudah OUT)
        # B 3333 OUT memiliki 2 transaksi (IN & OUT) yang sudah selesai
        deleted_count = db_manager.delete_completed_transactions(db_path=self.db_path)
        self.assertTrue(deleted_count >= 2)

        # Test clear all transactions
        cleared = db_manager.clear_all_transactions(db_path=self.db_path)
        self.assertTrue(cleared >= 1)
        self.assertEqual(len(db_manager.list_transactions(db_path=self.db_path)), 0)


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
