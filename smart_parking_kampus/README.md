# Smart Parking Kampus — Sistem Deteksi Plat Nomor Kendaraan

Proyek sistem parkir pintar berbasis kecerdasan buatan (*Computer Vision*) untuk lingkungan kampus menggunakan **YOLO (Deteksi Plat)**, **EasyOCR (Pembacaan Teks)**, dan **SQLite (Basis Data Registrasi & Verifikasi)**.

---

## 📌 Alur Kerja Sistem (Pipeline)

```text
Kamera / Gambar 
      │
      ▼
1. YOLO (Deteksi Kotak Plat)
      │
      ▼
2. Pemotongan (Crop) & Pra-pemrosesan Citra (CLAHE & Resize)
      │
      ▼
3. EasyOCR (Ekstraksi Karakter Alfanumerik)
      │
      ▼
4. Normalisasi Plat & Filter Kontekstual
      │
      ▼
5. Pencocokan Database SQLite Kampus
      │
      ▼
6. Verifikasi & Konfirmasi Petugas (Pencegahan Duplikasi & Koreksi Manual)
      │
      ▼
7. Penyimpanan Transaksi & Dashboard Riwayat
```

> **Prinsip Penting:**  
> - Deteksi dan OCR hanya bertugas membaca karakter plat nomor.  
> - SQLite adalah sumber resmi identitas pemilik dan kategori tarif yang dimasukkan oleh pihak kampus (bukan ditebak oleh model AI).  
> - Hasil OCR yang kurang yakin (`confidence < threshold`) tidak langsung dianggap benar.  
> - Plat yang tidak terdaftar akan berstatus **BELUM TERDAFTAR — PERLU VERIFIKASI PETUGAS**.  
> - Kecocokan nomor plat tidak membuktikan identitas fisik pengemudi dan tidak menjamin bahwa plat fisik kendaraan tidak dipalsukan.

---

## 💰 Struktur Kategori & Tarif Parkir

- **Dosen**: Rp0 (Gratis)
- **Mahasiswa**: Rp2.000
- **Tamu**: Rp5.000
- **Belum Terdaftar / OCR Kurang Yakin**: Minta verifikasi fisik oleh petugas parkir

---

## 🚀 Instalasi & Menjalankan Aplikasi (Windows PowerShell)

Pastikan virtual environment telah aktif:

```powershell
# Navigasi ke folder project
cd C:\Users\randr\Downloads\smart_parking_kampus_v2\smart_parking_kampus

# Pastikan pip mutakhir dan dependensi terpasang
..\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Menjalankan unit test
..\.venv\Scripts\python.exe test_helpers.py

# Menjalankan aplikasi Streamlit
..\.venv\Scripts\python.exe -m streamlit run app.py
```

Buka peramban di: **`http://localhost:8501`**

> **Catatan:** Semua fungsi pengelolaan database, ekspor/impor CSV, simulasi tarif, dashboard transaksi, serta uji OCR manual dapat berjalan langsung tanpa perlu menunggu file bobot model YOLO.

---

## 📂 Struktur Direktori Proyek

```text
smart_parking_kampus/
├── app.py                  # Antarmuka Streamlit (Kamera, Uji Manual, Database, Riwayat, Panduan)
├── parking_logic.py        # Logika normalisasi plat, disambiguasi karakter, dan penetapan tarif
├── db_manager.py           # Manajemen SQLite berparameter, migrasi tabel, deduplikasi, & metrik
├── ocr_engine.py           # Pra-pemrosesan citra plat (CLAHE, scaling) & ekstraksi EasyOCR
├── dataset_validator.py    # Validator integritas dataset YOLO, format anotasi, & deteksi kebocoran
├── train_model.py          # Script training YOLO mandiri dengan validasi dataset pra-training
├── test_helpers.py         # Unit tests (normalisasi, tarif, database, deduplikasi, validasi dataset)
├── parking.db              # Database SQLite lokal
├── dataset/                # Folder dataset plat nomor
│   ├── data.yaml           # Konfigurasi YOLO dataset
│   ├── images/
│   │   ├── train/          # Gambar training
│   │   └── val/            # Gambar validasi
│   └── labels/
│       ├── train/          # Label YOLO .txt
│       └── val/            # Label YOLO .txt
└── models/
    └── license_plate_detector.pt  # Lokasi bobot model YOLO terbaik (setelah training)
```

---

## 📋 PANDUAN INTEGRASI DATASET (Saat Dataset Dikirim)

Saat dataset plat nomor Anda siap dan dikirimkan, **sistem telah dilengkapi pipeline otomatis** untuk memeriksa dan memprosesnya:

### 1. Struktur Folder Dataset yang Diharapkan
Letakkan file dataset ke dalam direktori `dataset/`:
```text
dataset/
├── data.yaml
├── images/train/    <-- File citra (.jpg, .png, .jpeg)
├── images/val/
├── labels/train/    <-- File anotasi YOLO (.txt)
└── labels/val/
```

### 2. Format Anotasi YOLO
Setiap file gambar harus memiliki pasangan file teks `.txt` dengan nama yang sama:
```text
0 <x_center> <y_center> <width> <height>
```
- **Class ID**: Harus `0` (satu kelas: `license_plate`).
- **Koordinat**: Ternormalisasi dalam rentang `0.0` sampai `1.0`.

### 3. Pemeriksaan Otomatis (Validator)
Sebelum melakukan training, sistem akan memeriksa:
- Format file gambar (didukung: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
- Kesesuaian jumlah dan pasangan nama file gambar dan label.
- Kebocoran data (*data leakage*) antara split `train` dan `val`.
- Validitas bounding box (tidak boleh minus, tidak boleh `> 1.0`, tidak boleh `w <= 0` atau `h <= 0`).
- Label kosong atau file rusak.

Untuk menguji validitas dataset saja tanpa menjalankan training:
```powershell
..\.venv\Scripts\python.exe train_model.py --validate-only
```

*Jika dataset Anda ternyata menggunakan format berbeda (seperti Pascal VOC XML atau COCO JSON), validator akan mendeteksi dan kami akan menyiapkan konversi anotasi ke YOLO secara otomatis sebelum training.*

---

## 🏋️‍♂️ Melatih Model YOLO (Setelah Dataset Tersedia)

Setelah dataset lolos validasi, jalankan training dengan:
```powershell
..\.venv\Scripts\python.exe train_model.py --epochs 60 --batch 8 --auto-copy
```

**Parameter yang dapat disesuaikan:**
- `--model`: Bobot dasar YOLO (default: `yolo11n.pt`).
- `--epochs`: Jumlah putaran training (default: `60`).
- `--batch`: Ukuran batch (default: `8`).
- `--imgsz`: Ukuran resolusi citra input (default: `640`).
- `--auto-copy`: Menyalin otomatis bobot terbaik (`best.pt`) ke `models/license_plate_detector.pt`.

Setelah file `models/license_plate_detector.pt` tersedia, buka tab **Kamera & Verifikasi** di aplikasi Streamlit untuk memulai deteksi langsung via kamera.
