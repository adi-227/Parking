# 🅿️ Smart Parking Kampus — Sistem Deteksi Plat Nomor Kendaraan

Sistem otomasi gerbang parkir pintar berbasis kecerdasan buatan (*Computer Vision*) untuk lingkungan kampus menggunakan **YOLO (Deteksi Plat Nomor)**, **EasyOCR (Pembacaan Karakter Alfanumerik)**, dan **SQLite (Basis Data Registrasi & Verifikasi Hak Parkir)** dengan antarmuka modern **Streamlit**.

---

## 📌 Alur Kerja Sistem (Pipeline)

```text
Kamera / Gambar Kendaraan
      │
      ▼
1. YOLO (Deteksi Koordinat Kotak Plat Nomor)
      │
      ▼
2. Pemotongan (Crop) & Pra-pemrosesan Citra (CLAHE Contrast & Resize)
      │
      ▼
3. EasyOCR (Ekstraksi Karakter Alfanumerik Plat)
      │
      ▼
4. Normalisasi Plat & Disambiguasi Karakter Indonesia
      │
      ▼
5. Pencocokan Database SQLite Kampus (Civitas vs Tamu)
      │
      ▼
6. Mode Otomatis Cerdas: Deteksi Masuk (IN) vs Keluar (OUT) & Perhitungan Durasi
      │
      ▼
7. Verifikasi & Konfirmasi Petugas (Pencegahan Duplikasi & Audit Trail)
      │
      ▼
8. Penyimpanan Transaksi, Update Kuota Slot Parkir, & Dashboard Laporan
```

---

## 🌟 Fitur Utama Sistem

1. **Multi-Input Deteksi Plat:**
   - 📹 **Kamera Langsung (Webcam):** Pengambilan gambar real-time & deteksi otomatis.
   - 📁 **Unggah Foto Kendaraan:** Mendukung format JPG, JPEG, PNG.
   - ⌨️ **Ketik Plat Manual:** Fallback aman jika plat kotor/rusak.
2. **Mode Gerbang Cerdas (`Smart State Detection`):**
   - Sistem otomatis memeriksa status kendaraan di database: jika belum ada di dalam kampus maka otomatis **MASUK (`IN`)**, jika sudah di dalam maka otomatis **KELUAR (`OUT`)**.
3. **Pelacakan Durasi & Tarif Parkir:**
   - Menghitung durasi parkir menit per menit saat kendaraan keluar.
   - **Dosen:** Tarif Rp0 (Bebas Parkir).
   - **Mahasiswa:** Tarif Rp2.000.
   - **Tamu / Belum Terdaftar:** Tarif Rp5.000.
4. **Kapasitas Slot Parkir Real-Time:**
   - Visual progress bar dan indikator sisa kuota parkir otomatis di sidebar.
5. **Dashboard & Laporan Transaksi:**
   - Metrik riwayat, pencarian plat, filter kategori, serta ekspor laporan transaksi ke format **CSV**.
6. **Manajemen Data Kendaraan (CRUD):**
   - Tambah, ubah, hapus civitas kampus, serta fitur impor & ekspor massal via file CSV.

---

## 🚀 Panduan Menjalankan Aplikasi

### 1. Prasyarat
- Python 3.10 / 3.11 / 3.12
- Git

### 2. Kloning Repository
```bash
git clone git@github.com:adi-227/Parking.git
cd Parking
```

### 3. Buat Virtual Environment & Pasang Dependensi
```powershell
# Buat virtual environment
python -m venv .venv

# Aktifkan virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Masuk ke direktori aplikasi & pasang dependensi
cd smart_parking_kampus
pip install -r requirements.txt
```

### 4. Jalankan Unit Test
```powershell
python test_helpers.py
```

### 5. Jalankan Aplikasi Web
```powershell
streamlit run app.py
```
Akses aplikasi melalui browser: **`http://localhost:8501`**

---

## 📂 Struktur Direktori Proyek

```text
Parking/
├── .gitignore                      # Konfigurasi file yang diabaikan Git
├── README.md                       # Dokumentasi utama proyek
└── smart_parking_kampus/           # Folder aplikasi utama
    ├── app.py                      # Antarmuka Streamlit (Pos Gerbang, Data Master, Riwayat, Panduan)
    ├── parking_logic.py            # Logika normalisasi plat, disambiguasi, durasi, & penetapan tarif
    ├── db_manager.py               # Operasi SQLite (migrasi tabel, transaksi, kuota slot parkir)
    ├── ocr_engine.py               # Pra-pemrosesan gambar (CLAHE) & ekstraksi EasyOCR
    ├── dataset_validator.py        # Validasi format anotasi YOLO & dataset
    ├── train_model.py              # Skrip training mandiri YOLO
    ├── test_helpers.py             # 15 unit tests otomatis
    ├── parking.db                  # Database SQLite lokal
    ├── requirements.txt            # Daftar pustaka Python yang dibutuhkan
    ├── dataset/                    # Dataset plat nomor untuk pelatihan model
    │   ├── data.yaml
    │   ├── images/
    │   └── labels/
    └── models/
        └── license_plate_detector.pt  # Bobot model YOLO terlatih
```

---

## 🏋️ Pelatihan Ulang Model YOLO (Jika Ada Dataset Baru)

Untuk melatih ulang model deteksi plat nomor dengan gambar baru:
```powershell
cd smart_parking_kampus
python train_model.py --epochs 60 --batch 8 --auto-copy
```

---

## 📄 Lisensi & Kontributor
Dikembangkan untuk keperluan sistem manajemen parkir kampus berbasis *Computer Vision* & *Deep Learning*.
- Repository: [github.com/adi-227/Parking](https://github.com/adi-227/Parking)
