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
   - ⚡ **Kamera Cepat (Snapshot HTML5):** Pengambilan gambar instan 1-klik native browser (60 FPS, 0% lag, pemrosesan AI ~0.5–1.2 detik).
   - 📹 **Kamera Live (Streaming WebRTC):** Pelacakan kotak plat *real-time* 30 FPS dengan *anti-jitter* EMA smoothing & multi-sudut (lurus dan serong s.d 45°).
   - ⌨️ **Ketik Plat Manual:** Fallback aman jika plat kotor, rusak, atau terhalang.
2. **Mode Gerbang Cerdas (`Smart State Detection`):**
   - Sistem otomatis mengenali status kendaraan: jika belum ada di dalam area kampus otomatis berstatus **MASUK (`IN`)**, jika sudah di dalam otomatis berstatus **KELUAR (`OUT`)**.
3. **Pelacakan Durasi & Tarif Parkir Akurat (Tanpa Double Charge):**
   - **Gerbang Masuk (IN):** Dicatat **Rp0** (penerbitan tiket masuk).
   - **Gerbang Keluar (OUT):** Pembayaran dihitung berdasarkan selisih waktu masuk:
     - 👨‍🏫 **Dosen & Tendik:** Bebas biaya (Rp0).
     - 🎓 **Mahasiswa:** Tarif flat subsidi kampus **Rp2.000** (seharian).
     - 🚗 **Tamu / Pengunjung Luar:** 1 Jam Pertama (≤ 60 menit) **Rp3.000**, jam berikutnya **+Rp2.000 / jam**.
4. **Kapasitas Slot Parkir Real-Time:**
   - Indikator visual progress bar dan kuota sisa slot parkir otomatis di sidebar pos gerbang.
5. **Dashboard & Laporan Transaksi Lengkap:**
   - Metrik pemeriksaan, pemisahan statistik civitas vs tamu, riwayat transaksi gerbang, filter arah (IN/OUT), serta ekspor laporan ke format **CSV**.
6. **Manajemen Kendaraan & Sesi Parkir (CRUD Lengkap):**
   - CRUD data master civitas kampus (Dosen & Mahasiswa) + impor/ekspor CSV massal.
   - CRUD sesi parkir aktif (melihat durasi berjalan, manual entry, edit data sesi, dan checkout/hapus per kendaraan).

---

## 🛠️ Teknologi & Tools yang Digunakan

| Kategori | Tool / Library | Versi | Fungsi & Kegunaan |
|---|---|---|---|
| **Framework Web & UI** | [Streamlit](https://streamlit.io/) | `v1.36+` | Dashboard interaktif web, kontrol sidebar pos gerbang, visualisasi metrik, dan tabel CRUD. |
| **Object Detection (AI)** | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | `v8.2+` | Deteksi posisi koordinat kotak plat nomor kendaraan (*bounding box*) secara presisi dan cepat. |
| **Optical Character Recognition** | [EasyOCR](https://github.com/JaidedAI/EasyOCR) | `v1.7+` | Ekstraksi dan pengenalan karakter alfanumerik plat nomor dari gambar crop. |
| **Computer Vision & Processing** | [OpenCV (cv2)](https://opencv.org/) & [NumPy](https://numpy.org/) | `v4.9+` / `v1.26+` | Pra-pemrosesan citra (CLAHE contrast enhancer, crop, grayscale, geometric filtering, EMA box smoothing). |
| **Kamera & Streaming Video** | [Streamlit-WebRTC](https://github.com/whitphx/streamlit-webrtc) | `v0.47+` | Protokol streaming video WebRTC *low-latency* dari webcam untuk bidik plat dinamis. |
| **Basis Data (Database)** | [SQLite3](https://www.sqlite.org/) | Built-in | Database relasional lokal (`parking.db`) untuk data master kendaraan, sesi aktif, dan audit transaksi. |
| **Manipulasi Data & CSV** | [Pandas](https://pandas.pydata.org/) | `v2.0+` | Agregasi metrik laporan, manipulasi data tabular, serta fungsi ekspor/impor file CSV. |
| **Testing Otomatis** | Python `unittest` | Built-in | 27 skenario unit test otomatis (`test_helpers.py`) untuk memverifikasi logika tarif, deteksi, dan database. |
| **Akses Publik (Tunnelling)** | [Localtunnel](https://localtunnel.github.io/www/) | CLI | Port forwarding instan untuk membuka akses Streamlit pos gerbang ke internet via HTTPS. |

---

## ⏱️ Rincian Waktu Deteksi & Capture Kamera

| Tahapan Proses | Durasi Waktu | Keterangan Teknis |
|---|---|---|
| **1. Pengambilan Gambar (Capture)** | **< 0.1 Detik** *(Instan)* | Pengambilan frame langsung melalui browser HTML5 Canvas / WebRTC frame buffer. |
| **2. Deteksi Objek (YOLOv8)** | **~0.05 – 0.15 Detik** *(50–150 ms)* | Inference model custom `license_plate_detector.pt` pada resolusi gambar 640×480. |
| **3. Pra-Pemrosesan Citra** | **~0.01 Detik** *(10 ms)* | Pemotongan koordinat plat (*crop*), konversi warna, dan peningkatan kontras adaptif CLAHE. |
| **4. Pembacaan Karakter (EasyOCR)** | **~0.30 – 0.80 Detik** *(300–800 ms)* | Ekstraksi karakter teks alfanumerik huruf dan angka pada plat nomor kendaraan. |
| **5. Validasi & Disambiguasi** | **< 0.005 Detik** *(1–5 ms)* | Koreksi otomatis karakter tertukar (I/1, O/0), validasi pola plat Indonesia, & query SQLite. |
| **TOTAL WAKTU SCAN (KLIK HINGGA TIKET MUNCUL)** | **⚡ ~0.5 – 1.2 Detik** | Sangat responsif untuk operasional pos gerbang masuk/keluar kendaraan secara cepat. |
| **Streaming Real-Time FPS** | **30 FPS** | Interval deteksi live box per **3 frame (~100 ms)** dengan pelacak *anti-jitter*. |
| **Cooldown Duplikasi Scan** | **60 Detik** *(Default)* | Mencegah kendaraan yang sama tercatat ganda berturut-turut pada arah yang sama. |

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
    ├── test_helpers.py             # 27 unit tests otomatis (tarif, OCR, IoU, tracker, DB)
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

