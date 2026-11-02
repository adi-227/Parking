# DRAF PROPOSAL / LAPORAN TUGAS AKHIR (BAB I — BAB III)

**Program Studi:** Informatika (S1)  
**Fakultas:** Teknik dan Informatika — Universitas Bina Sarana Informatika  
**Topik Penelitian:** Sistem Deteksi Plat Nomor Kendaraan (*Automatic License Plate Recognition*) untuk Otomasi Gerbang Parkir Kampus

---

## BAB I: PENDAHULUAN

### 1.1. Latar Belakang Masalah
Peningkatan volume kepemilikan kendaraan bermotor, baik roda dua maupun roda empat, di lingkungan perguruan tinggi menuntut tata kelola area parkir yang tertib, cepat, dan transparan. Pada area kampus, pengguna parkir terbagi menjadi dua kelompok utama, yaitu civitas akademika (dosen, tenaga kependidikan, dan mahasiswa) serta masyarakat umum atau tamu dari luar kampus. Masing-masing kategori memiliki hak akses dan skema tarif yang berbeda: dosen dan staf umumnya dibebaskan dari biaya (gratis), mahasiswa memperoleh tarif flat subsidi kampus, sedangkan tamu dikenakan tarif progresif berdasarkan durasi waktu parkir.

Namun, implementasi sistem perparkiran konvensional yang saat ini banyak beroperasi masih menghadapi sejumlah kendala signifikan:
1. **Ketergantungan pada Karcis Kertas dan Kartu Manual:** Penggunaan tiket fisik rawan hilang atau rusak akibat basah saat hujan. Kehilangan tiket memicu perselisihan di gerbang keluar, menimbulkan denda administratif, dan memperlambat arus antrean kendaraan.
2. **Keterbatasan Verifikasi Manual Hak Akses:** Petugas gerbang pos parkir tidak dapat memverifikasi keabsahan identitas pengendara secara instan, sehingga berisiko terjadi penyalahgunaan hak tarif civitas oleh pihak luar.
3. **Pencatatan Finansial dan Durasi yang Rawan Galat (*Human Error*):** Perhitungan waktu masuk dan keluar secara manual sering menghasilkan selisih hitung tagihan, ketidakkonsistenan pembulatan jam parkir, serta celah kebocoran pendapatan retribusi parkir.
4. **Ketiadaan Monitoring Kapasitas Lahan Real-Time:** Pengendara tidak memperoleh informasi ketersediaan slot kosong sebelum melewati gerbang, yang sering kali berujung pada kepadatan sirkulasi di dalam area parkir.

Perkembangan teknologi *Artificial Intelligence* (AI), khususnya di bidang *Computer Vision* dan *Deep Learning*, menawarkan solusi melalui sistem *Automatic License Plate Recognition* (ALPR). Dengan mendayagunakan arsitektur **YOLOv8 (*You Only Look Once version 8*)** untuk melokalisasi kotak plat nomor secara presisi tinggi dan **EasyOCR** untuk mengekstrak string alfanumerik, sistem dapat mengenali identitas kendaraan secara nirkontak (*contactless*). Selain itu, integrasi basis data relasional **SQLite** dan antarmuka web interaktif **Streamlit** memungkinkan sistem melakukan verifikasi hak akses civitas vs tamu, menghitung tarif progresif secara deterministik, serta menyajikan visualisasi data transaksi secara *real-time*.

Oleh karena itu, penelitian ini berfokus pada rancang bangun sistem parkir pintar kampus berbasis *Computer Vision* yang andal terhadap variasi sudut kamera, pencahayaan luar ruangan, dan foto jarak dekat (*close-up*).

---

### 1.2. Rumusan Masalah
Berdasarkan latar belakang di atas, rumusan masalah dalam penelitian ini dirumuskan sebagai berikut:
1. Bagaimana mengimplementasikan model *Deep Learning* YOLOv8 dan EasyOCR untuk mendeteksi serta membaca plat nomor kendaraan standar Indonesia secara akurat dari sudut lurus maupun serong (*angle*) hingga 45°?
2. Bagaimana merancang pipeline prapemrosesan citra digital guna mereduksi derau (*noise*), ketimpangan kontras pencahayaan, dan artefak kamera sebelum dilakukan pengenalan karakter?
3. Bagaimana membangun mekanisme pelacakan temporal dan filter geometri kotak plat untuk mengeliminasi deteksi palsu (*false positive*) dari objek non-plat (seperti wajah, tangan, atau logo pakaian)?
4. Bagaimana merancang model logika penentuan status gerbang (*Smart State Detection*: IN vs OUT) dan formulasi tarif progresif yang adil tanpa risiko duplikasi penagihan (*double charge*)?

---

### 1.3. Tujuan dan Manfaat

#### 1.3.1. Tujuan Penelitian
Tujuan yang hendak dicapai dalam penelitian ini adalah:
1. Merancang dan membangun arsitektur perangkat lunak *Smart Parking System* berbasis web menggunakan Python dan Streamlit.
2. Menerapkan tahapan prapemrosesan citra digital (*Grayscale, CLAHE Contrast Enhancement, Gaussian Blur Denoising, dan Hough Transform Deskewing*) untuk meningkatkan akurasi pembacaan karakter alfanumerik pada EasyOCR.
3. Mengembangkan algoritma penapisan geometri (*Aspect Ratio*) dan pelacak temporal *Exponential Moving Average* (EMA) guna menjaga kestabilan *bounding box* kamera *real-time*.
4. Mengintegrasikan basis data relasional SQLite untuk pengelolaan data master civitas, sesi parkir aktif, dan pencatatan audit transaksi masuk/keluar gerbang secara otomatis.

#### 1.3.2. Manfaat Penelitian
Penelitian ini diharapkan memberikan manfaat sebagai berikut:
1. **Manfaat Teoritis (Akademis):** Memberikan kontribusi keilmuan dalam penerapan praktis integrasi model *Object Detection* mutakhir (YOLOv8) dengan mesin OCR berbasis *Convolutional Recurrent Neural Network* (CRNN) pada domain citra plat nomor kendaraan Indonesia.
2. **Manfaat Praktis (Operasional Kampus):** Membantu pengelola fasilitas kampus mempercepat laju sirkulasi kendaraan di gerbang masuk/keluar, menghilangkan biaya cetak karcis fisik, serta menjamin transparansi pencatatan pendapatan retribusi parkir secara otomatis.
3. **Manfaat bagi Pengguna (Civitas & Tamu):** Memberikan kenyamanan akses parkir yang cepat tanpa repot mengambil karcis fisik, serta menjamin perhitungan biaya parkir yang akurat dan transparan sesuai durasi riil.

---

### 1.4. Hipotesis
Hipotesis yang diajukan dalam penelitian ini adalah:
* **Hipotesis Nol ($H_0$):**  
  Penerapan kombinasi model YOLOv8, pipeline prapemrosesan citra (CLAHE dan Gaussian Denoising), serta pelacak EMA tidak memberikan peningkatan akurasi deteksi plat nomor dan tidak mampu menghasilkan waktu pemrosesan (*inference latency*) di bawah 1,5 detik per kendaraan.
* **Hipotesis Alternatif ($H_a$):**  
  Penerapan kombinasi model YOLOv8, pipeline prapemrosesan citra (CLAHE dan Gaussian Denoising), serta pelacak EMA mampu mendeteksi plat nomor kendaraan secara akurat dari berbagai sudut pengambilan gambar (lurus maupun serong hingga 45°), menghasilkan nilai *Mean Average Precision* ($\text{mAP@0.5}$) $\ge 90\%$, menekan *Character Error Rate* (CER) pada pembacaan teks $\le 10\%$, dan mencapai total waktu pemrosesan rata-rata $\le 1,2$ detik per transaksi.

---

### 1.5. Batasan Masalah
Agar penelitian tetap terfokus dan mendalam, batasan masalah ditentukan sebagai berikut:
1. Objek penelitian dibatasi pada kendaraan bermotor roda dua (sepeda motor) dan roda empat (mobil) dengan plat nomor standar Republik Indonesia (kombinasi 1–2 huruf wilayah, 1–4 angka nomor polisi, dan 0–3 huruf seri).
2. Model pendeteksi objek menggunakan arsitektur YOLOv8 yang telah dilatih (*custom trained*) pada dataset plat nomor kendaraan.
3. Modul ekstraksi teks menggunakan pustaka EasyOCR berbasis *Deep Learning* dengan model karakter alfanumerik Latin.
4. Input citra bersumber dari webcam/kamera gerbang melalui dua mode: *Snapshot 1-Klik* (Kamera Cepat HTML5) dan *Streaming Real-Time* (WebRTC).
5. Pengujian dilakukan pada lingkungan pos parkir kampus dengan asumsi plat nomor tidak tertutup lumpur pekat atau rusak secara fisik melebihi 50%.
6. Basis data yang digunakan adalah SQLite lokal (`parking.db`).

---

## BAB II: LANDASAN TEORI

### 2.1. Tinjauan Pustaka

#### 2.1.1. Konsep Sistem Parkir Pintar (*Smart Parking System*)
Sistem parkir pintar adalah integrasi teknologi sensor, komputasi visual, dan basis data yang dirancang untuk mengotomatisasi pengawasan, perizinan akses, serta transaksi tarif parkir. Pada arsitektur cerdas modern, identifikasi kendaraan dilakukan menggunakan teknologi *Automatic License Plate Recognition* (ALPR) yang menggantikan tiket karcis manual guna meminimalkan interaksi fisik dan mempercepat arus sirkulasi gerbang.

#### 2.1.2. Citra Digital dan Fenomena Derau (*Noise*)
Citra digital yang ditangkap oleh kamera pada pos parkir luar ruangan (*outdoor*) sering kali mengalami degradasi akibat derau (*noise*). Derau merupakan fluktuasi acak pada nilai intensitas piksel citra yang tidak merepresentasikan informasi asli objek. 

Pada sistem ALPR, derau umumnya timbul akibat:
* **Derau Termal / Sensor (*Gaussian Noise*):** Muncul saat sensor kamera bekerja pada kondisi pencahayaan rendah (sore atau malam hari) dengan nilai ISO tinggi, menghasilkan bintik-bintik butiran pada permukaan plat nomor.
* **Derau Impuls (*Salt-and-Pepper Noise*):** Piksel putih atau hitam ekstrem akibat gangguan transmisi sinyal kamera.
* **Kabur Gerakan (*Motion Blur*):** Penurunan ketajaman tepi karakter akibat laju kendaraan saat melewati jangkauan lensa kamera.

Keberadaan derau ini menyebabkan tepi karakter plat menjadi tidak tegas, sehingga menurunkan akurasi pembacaan pada mesin OCR jika tidak dilakukan reduksi terlebih dahulu.

#### 2.1.3. Prapemrosesan Citra Digital dan Formulasi Matematis Reduksi Derau

##### 1. Konversi Skala Keabuan (*Grayscale Conversion*)
Citra warna RGB 3-kanal dikonversi menjadi citra intensitas 1-kanal untuk menyederhanakan komputasi dengan rumus bobot luminansi standar ITU-R BT.601:
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
di mana $R, G, B \in [0, 255]$ adalah komponen warna merah, hijau, dan biru, serta $Y$ adalah intensitas derajat keabuan.

##### 2. Reduksi Derau dengan Filter Gaussian (*Gaussian Denoising Filter*)
Untuk mereduksi derau acak frekuensi tinggi tanpa menghilangkan kontinuitas garis tepi huruf pada plat nomor, digunakan konvolusi spasial dengan fungsi distribusi Gaussian 2 dimensi:
$$G(x, y) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
Pada implementasi sistem, digunakan kernel diskrit berukuran $3 \times 3$ piksel dengan matriks bobot ternormalisasi:
$$K = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$
Nilai piksel baru $I_{denoised}(x, y)$ diperoleh melalui operasi konvolusi:
$$I_{denoised}(x, y) = \sum_{i=-1}^{1} \sum_{j=-1}^{1} I(x+i, y+j) \cdot K(i+1, j+1)$$

##### 3. Peningkatan Kontras Adaptif (*CLAHE*)
Metode *Contrast Limited Adaptive Histogram Equalization* (CLAHE) mengatasi pencahayaan tidak merata dengan membagi citra ke dalam matriks blok kontekstual lokal berukuran $8 \times 8$ piksel. Histogram lokal pada tiap blok dihitung dan dibatasi oleh ambang batas pemotongan (*clip limit* $\beta = 2.2$):
$$\beta = \frac{M \times N}{L} \left(1 + \frac{\alpha}{100} (S_{max} - 1)\right)$$
Piksel yang terpotong diredistribusikan secara seragam ke seluruh *bin* histogram sebelum dilakukan transformasi perataan kumulatif (*Cumulative Distribution Function* / CDF), sehingga kontras teks plat meningkat tajam tanpa memicu penguatan derau latar belakang.

##### 4. Deteksi Tepi dan Koreksi Sudut Kemiringan (*Deskewing*)
Kamera pos parkir yang terpasang di tiang samping gerbang menangkap citra plat nomor dengan sudut kemiringan tertentu. Sistem mendeteksi garis menggunakan Transformasi Hough Probabilistik pada tepi Canny:
$$\rho = x \cos \theta + y \sin \theta$$
Kemiringan garis dominan dihitung berdasarkan gradien koordinat:
$$\theta_i = \arctan\left(\frac{y_2 - y_1}{x_2 - x_1}\right) \times \frac{180^\circ}{\pi}$$
Jika nilai median kemiringan $|\theta_{median}| \ge 2.0^\circ$, citra plat nomor diputar kembali secara otomatis menggunakan matriks transformasi afina 2D:
$$M = \begin{bmatrix} \alpha & \beta & (1-\alpha) \cdot x_c - \beta \cdot y_c \\ -\beta & \alpha & \beta \cdot x_c + (1-\alpha) \cdot y_c \end{bmatrix}$$
di mana $\alpha = \cos(\theta_{median})$, $\beta = \sin(\theta_{median})$, dan $(x_c, y_c)$ adalah titik pusat rotasi citra.

---

#### 2.1.4. Deteksi Objek dengan YOLOv8 dan Verifikasi Geometris
YOLOv8 (*Ultralytics*) menerapkan pendekatan konvolusi *anchor-free* yang mendeteksi kotak pembatas (*bounding box*) $B = [x_1, y_1, x_2, y_2]$ dan kelas objek secara simultan dalam satu lintasan inferensi (*single-pass*).

##### 1. Formulasi *Intersection over Union* (IoU)
IoU mengukur rasio tumpang tindih antara kotak kandidat hasil prediksi ($B_p$) dan kotak acuan (*ground truth* $B_g$):
$$\text{IoU}(B_p, B_g) = \frac{\text{Area}(B_p \cap B_g)}{\text{Area}(B_p \cup B_g)} = \frac{\text{Area}(B_p \cap B_g)}{\text{Area}(B_p) + \text{Area}(B_g) - \text{Area}(B_p \cap B_g)}$$

##### 2. Verifikasi Rasio Aspek (*Aspect Ratio Filtering*)
Untuk mengeliminasi deteksi palsu (*false positive*) dari objek non-plat (seperti telapak tangan vertikal dengan rasio aspek $\sim 0.73$ atau wajah manusia dengan rasio $\sim 0.94$), sistem memverifikasi rasio aspek ($AR$) dan batas tinggi frame ($HR$):
$$AR = \frac{w}{h} = \frac{x_2 - x_1}{y_2 - y_1}$$
$$HR = \frac{h}{H_{frame}} = \frac{y_2 - y_1}{H_{frame}}$$
Kotak kandidat dinyatakan sah secara geometris jika memenuhi fungsi predikat:
$$\text{Valid}(B) = \left( 1.10 \le AR \le 5.80 \right) \;\land\; \left( HR \le 0.88 \right) \;\land\; (w \ge 35\text{ px}) \;\land\; (h \ge 15\text{ px})$$
Batas rasio tinggi maksimal $HR \le 0.88$ memungkinkan sistem tetap mengenali plat yang diambil dari jarak sangat dekat (*close-up*).

##### 3. Metrik Evaluasi Kinerja Deteksi
Kinerja model deteksi diukur melalui parameter:
$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$F_1\text{-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
$$\text{mAP@0.5} = \frac{1}{N} \sum_{k=1}^{N} \int_{0}^{1} P_k(R) \, dR$$
di mana $TP$ adalah jumlah plat terdeteksi benar ($\text{IoU} \ge 0.50$), $FP$ adalah deteksi salah pada objek non-plat, dan $FN$ adalah plat nomor yang gagal terdeteksi.

---

#### 2.1.5. Pelacakan Temporal Kotak (*Anti-Jitter* EMA Tracker)
Pada transmisi video *live streaming* 30 FPS, koordinat kotak deteksi sering kali bergetar (*jitter*). Sistem menerapkan *Exponential Moving Average* (EMA) dengan koefisien pemulusan $\alpha = 0.65$:
$$B_t^{smoothed} = \alpha \cdot B_t^{raw} + (1 - \alpha) \cdot B_{t-1}^{smoothed}$$
Kandidat kotak baru hanya akan diperbarui jika memiliki korelasi spasial $\text{IoU}(B_{t-1}, B_t) > 0.15$. Jika tidak, sistem mempertahankan posisi kotak sebelumnya hingga 6 *frame* (*grace period*) guna mencegah kotak melompat liar (*anti-jumping*).

---

#### 2.1.6. Pengenalan Karakter Alfanumerik (EasyOCR) dan Evaluasi CER
EasyOCR mengombinasikan *Convolutional Neural Network* (CNN) untuk ekstraksi fitur visual karakter, *Recurrent Neural Network* (RNN / Bi-LSTM) untuk memodelkan urutan huruf/angka, dan fungsi *Connectionist Temporal Classification* (CTC Loss) untuk menerjemahkan deretan sekuens menjadi teks.

Akurasi teks yang diekstraksi diukur menggunakan *Character Error Rate* (CER) berbasis Jarak Levenshtein (*Edit Distance*):
$$\text{CER} = \frac{S + D + I}{N_{ref}} \times 100\%$$
di mana:
* $S$: Jumlah substitusi karakter (misal karakter `B` terbaca `8`).
* $D$: Jumlah karakter terhapus/hilang (*deletions*).
* $I$: Jumlah karakter sisipan/palsu (*insertions*).
* $N_{ref}$: Total jumlah karakter pada plat nomor sebenarnya (*ground truth*).

Akurasi pembacaan karakter didefinisikan sebagai:
$$\text{Akurasi OCR} = 100\% - \text{CER}$$

---

#### 2.1.7. Formulasi Logika Bisnis & Perhitungan Tarif Parkir
Biaya parkir dihitung secara deterministik berdasarkan kategori kendaraan dan selisih waktu transaksi.

##### 1. Perhitungan Durasi Parkir ($\Delta t$)
Jika waktu masuk gerbang tercatat $t_{in}$ dan waktu keluar adalah $t_{out}$:
$$\Delta t_{menit} = \max\left(0, \; \left\lfloor \frac{t_{out} - t_{in}}{60\text{ detik}} \right\rfloor \right)$$

##### 2. Fungsi Matematis Tarif Parkir ($C$)
Biaya parkir dihitung berdasarkan fungsi bertingkat (*piecewise function*):
$$C(kategori, \Delta t) = \begin{cases} 
0, & \text{jika } kategori = \text{'Dosen'} \\ 
2.000, & \text{jika } kategori = \text{'Mahasiswa'} \\ 
3.000, & \text{jika } kategori = \text{'Tamu'} \land \Delta t \le 60 \\ 
3.000 + \left\lfloor \frac{\Delta t - 1}{60} \right\rfloor \times 2.000, & \text{jika } kategori = \text{'Tamu'} \land \Delta t > 60 
\end{cases}$$

Dengan aturan ini:
* Tamu yang parkir selama 1 menit dikenakan $\text{Rp3.000}$ (tarif jam pertama).
* Tamu yang parkir selama 75 menit dikenakan $\text{Rp3.000} + (1 \times \text{Rp2.000}) = \text{Rp5.000}$.
* Seluruh transaksi gerbang masuk (IN) dicatat dengan nominal $\text{Rp0}$ (penerbitan tiket masuk digital) dan baru ditagihkan saat melewati gerbang keluar (OUT).

---

### 2.2. Penelitian Terkait
Sebagai landasan perbandingan dalam perancangan sistem ini, berikut adalah pemetaan beberapa penelitian terdahulu yang relevan:

| No | Peneliti & Tahun | Judul Penelitian | Metode yang Digunakan | Kelebihan | Kelemahan / Celah Penelitian | Solusi pada Penelitian Ini |
|:--:|---|---|---|---|---|---|
| 1 | Rahmad dkk. (2021) | *Implementasi ALPR Menggunakan Haar Cascade dan Tesseract OCR* | Haar Cascade + Tesseract OCR | Komputasi ringan pada CPU standar | Akurasi anjlok drastis pada sudut serong ($>15^\circ$) dan plat kotor | Mengganti Haar Cascade dengan **YOLOv8** dan Tesseract dengan **EasyOCR + CLAHE** |
| 2 | Pratama & Sari (2022) | *Sistem Otomasi Gerbang Parkir Berbasis Kartu RFID dan Sensor Ultrasonik* | Mikrokontroler Arduino + RFID Card | Biaya komponen terjangkau, pembacaan kartu cepat | Pengguna tetap harus membuka jendela dan rawan kehilangan kartu RFID | Menggunakan **kamera cerdas nirkontak (*contactless*)** berbasis pengenalan plat nomor otomatis |
| 3 | Gunawan & Wibowo (2023) | *Deteksi Plat Nomor Kendaraan Menggunakan YOLOv5 dan OCR Berbasis Web* | YOLOv5 + PyTesseract + Flask | Mampu mendeteksi posisi plat nomor mobil dengan baik | Tidak ada filter geometri plat nomor motor, sering mendeteksi wajah dan tangan | Menerapkan **filter rasio aspek geometris ($1.10 \le AR \le 5.80$)** dan penyesuaian dudukan plat motor |
| 4 | Santoso dkk. (2023) | *Sistem Pengenalan Plat Nomor Menggunakan YOLOv7 pada Kamera CCTV* | YOLOv7 + EasyOCR | Akurasi deteksi tinggi pada kondisi siang hari | Bounding box bergetar (*jittering*) pada video live dan latensi tinggi | Menerapkan **EMA Smoothing ($\alpha=0.65$)**, spatial IoU locking, dan Snapshot 1-Klik |
| 5 | Kusuma & Hidayat (2024) | *Rancang Bangun Smart Parking Kampus dengan Perhitungan Tarif Progresif* | Barcode Scanner + Database MySQL | Perhitungan tarif terstruktur rapi | Masih menggunakan tiket kertas cetak yang menimbulkan limbah | Mengintegrasikan **database SQLite lokal terotomasi** langsung dengan pengenalan plat ALPR |

---

## BAB III: METODOLOGI PENELITIAN

### 3.1. Proses dan Langkah Penelitian
Penelitian ini disusun menggunakan model rekayasa perangkat lunak terstruktur (*Prototyping / Waterfall Model*) yang mencakup 6 (enam) tahapan utama sebagaimana diilustrasikan pada diagram alir berikut:

```
┌────────────────────────────────────────────────────────┐
│ 1. Identifikasi Masalah & Studi Literatur              │
│    - Mengkaji kendala antrean, tiket hilang, & tarif   │
│    - Studi pustaka YOLOv8, EasyOCR, CLAHE, & SQLite    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 2. Pengumpulan Dataset & Pra-pemrosesan Citra          │
│    - Pengambilan sampel foto plat motor dan mobil      │
│    - Pelabelan Bounding Box (Anotasi Format YOLO)      │
│    - Augmentasi: variasi rotasi serong, kontras, & blur│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 3. Perancangan Arsitektur Sistem & Basis Data          │
│    - Pipeline deteksi, cropping, denoising, & OCR      │
│    - Skema database relasional (vehicles, transactions)│
│    - Perancangan UI Pos Gerbang berbasis Streamlit     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 4. Implementasi Perangkat Lunak                        │
│    - Pelatihan/konfigurasi model YOLOv8 (PT weight)    │
│    - Implementasi Denoising CLAHE & Hough Transform    │
│    - Implementasi Tracking EMA & Rule Engine Tarif     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 5. Pengujian Sistem Terstruktur (Rencana Bab IV)       │
│    - Uji Unit Logic (27 skenario test_helpers.py)      │
│    - Uji Akurasi Deteksi Plat (IoU, Precision, Recall) │
│    - Uji Akurasi OCR (CER) pada plat lurus vs serong   │
│    - Uji Waktu Tanggap / Latensi Inferensi Sistem      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 6. Analisis Hasil, Evaluasi, & Dokumentasi             │
│    - Penarikan kesimpulan berdasarkan pengujian        │
│    - Penyusunan laporan akhir tugas akhir              │
└────────────────────────────────────────────────────────┘
```

#### Rincian Langkah Penelitian:
1. **Identifikasi Masalah & Studi Literatur:** Mengidentifikasi kendala pada tata kelola parkir kampus, mengkaji arsitektur model deteksi objek modern YOLOv8, algoritma prapemrosesan citra digital, serta mekanisme pengenalan teks EasyOCR.
2. **Pengumpulan Dataset:** Menghimpun sampel citra kendaraan bermotor (mobil dan motor) yang mencakup plat nomor Indonesia dari sudut lurus, sudut serong hingga 45°, serta variasi pencahayaan luar ruangan.
3. **Perancangan Arsitektur:** Merancang diagram aliran data dari penangkapan citra oleh kamera, pelokalisasian plat oleh YOLOv8, prapemrosesan pembersihan derau, ekstraksi alfanumerik oleh OCR, hingga query ke basis data SQLite.
4. **Implementasi Sistem:** Mengonstruksi modul program Python yang terdiri dari `ocr_engine.py` (filter citra dan OCR), `parking_logic.py` (rule engine tarif dan validasi plat), `db_manager.py` (basis data relasional), dan `app.py` (antarmuka web Streamlit).
5. **Rencana Pengujian:** Melakukan verifikasi komprehensif yang dirancang untuk dieksekusi dan dievaluasi pada Bab IV, mencakup pengujian unit otomatis, uji kinerja deteksi objek, uji pembacaan teks plat, serta uji latensi waktu nyata.
6. **Dokumentasi & Evaluasi:** Menganalisis hasil pengujian terhadap hipotesis penelitian dan menyusun rekomendasi pengembangan.

---

### 3.2. Metode Pengolahan dan Analisis Data

#### 3.2.1. Metode Pengolahan Citra (*Image Processing Pipeline*)
Data citra masukan dari kamera gerbang diproses melalui alur bertingkat:
1. **Akuisisi Citra:** Gambar ditangkap pada resolusi $640 \times 480$ piksel melalui modul *HTML5 Camera* atau *WebRTC Frame Buffer*.
2. **Inferensi YOLOv8:** Citra dievaluasi untuk mendeteksi koordinat *bounding box* plat $[x_1, y_1, x_2, y_2]$ dengan ambang batas *confidence* minimal $0.35$.
3. **Validasi Geometri & Rasio:** Kotak yang lolos harus memenuhi rasio aspek $1.10 \le AR \le 5.80$ dan rasio tinggi $HR \le 0.88$. Jika kotak YOLO tidak terdeteksi (misal pada foto sangat *close-up*), sistem mengaktifkan mekanisme *Smart Fallback* dengan melakukan analisis langsung pada citra utuh dan area tengah (*center-crop*).
4. **Prapemrosesan Plat (*Cropped Patch*):**
   * Penskalaan ulang proporsional (*bicubic interpolation*) jika tinggi plat $< 110\text{ px}$.
   * Konversi ke citra keabuan (*Grayscale*).
   * Peningkatan kontras adaptif (*CLAHE*) dengan *tile grid* $8 \times 8$ dan *clip limit* $2.2$.
   * Penapisan derau frekuensi tinggi menggunakan *Gaussian Blur* kernel $3 \times 3$.
   * Koreksi kemiringan rotasi menggunakan Transformasi Hough jika ditemukan sudut median $|\theta| \ge 2.0^\circ$.
5. **Ekstraksi Teks (EasyOCR):** Pembacaan karakter alfanumerik dari citra hasil prapemrosesan.
6. **Normalisasi & Disambiguasi Karakter:** Teks hasil OCR dibersihkan menggunakan ekspresi reguler (*regex*) pola plat Indonesia `^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$`. Sistem memetakan karakter yang sering tertukar berdasarkan konteks letak (misal karakter `O` diubah menjadi angka `0` pada posisi nomor polisi tengah, dan angka `1` diubah menjadi huruf `I` atau `B` pada posisi kode wilayah depan).

#### 3.2.2. Metode Analisis dan Evaluasi Data (Rencana Pengujian Bab IV)
Untuk membuktikan hipotesis penelitian ($H_a$), data hasil pengujian yang akan dianalisis pada Bab IV dikelompokkan ke dalam 4 (empat) parameter uji:

##### 1. Analisis Kinerja Model Deteksi Objek (YOLOv8)
Kinerja model pendeteksi plat nomor dievaluasi menggunakan matriks konfusi (*Confusion Matrix*) dengan menghitung:
* Nilai *Precision*, *Recall*, dan $F_1$-Score pada ambang batas $\text{IoU} \ge 0.50$.
* Nilai *Mean Average Precision* ($\text{mAP@0.5}$ dan $\text{mAP@0.5:0.95}$) untuk mengukur ketepatan lokalisasi kotak plat di berbagai sudut pengambilan gambar.

##### 2. Analisis Akurasi Pengenalan Karakter (EasyOCR)
Akurasi pembacaan teks plat nomor dievaluasi dengan membandingkan teks prediksi sistem terhadap teks plat nomor aktual (*ground truth*):
$$\text{CER} = \frac{S + D + I}{N_{ref}} \times 100\%$$
Pengujian dilakukan pada 3 (tiga) kategori skenario:
* Skenario A: Plat nomor posisi tegak lurus (jarak $1 - 2$ meter).
* Skenario B: Plat nomor posisi serong ($15^\circ - 45^\circ$).
* Skenario C: Plat nomor jarak dekat (*close-up*, tinggi plat $> 50\%$ frame).

##### 3. Analisis Latensi dan Waktu Pemrosesan Sistem
Mengukur waktu tanggap sistem (*end-to-end latency*) dari saat tombol ambil foto ditekan hingga informasi tiket parkir ditampilkan di layar Streamlit:
$$T_{total} = T_{capture} + T_{yolo} + T_{preprocess} + T_{ocr} + T_{db\_query}$$
Target analisis adalah memastikan $T_{total} \le 1.2\text{ detik}$.

##### 4. Analisis Verifikasi Logika Bisnis & Pengujian Unit
Menguji keandalan logika bisnis transaksi dan penentuan tarif parkir menggunakan 27 skenario pengujian unit otomatis (*Unit Test*) pada berkas `test_helpers.py`, meliputi:
* Validasi tarif tamu 1 jam pertama ($\text{Rp3.000}$) dan jam tambahan progresif ($+\text{Rp2.000}$/jam).
* Validasi tarif subsidi flat mahasiswa ($\text{Rp2.000}$) dan gratis dosen ($\text{Rp0}$).
* Validasi pemblokiran duplikasi transaksi (*cooldown idempotency* 60 detik).
* Validasi penolakan objek non-plat (wajah, pakaian, telapak tangan).
