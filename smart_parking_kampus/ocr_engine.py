import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_plate_image(image: np.ndarray) -> np.ndarray:
    """
    Pra-pemrosesan gambar plat nomor sebelum EasyOCR untuk meningkatkan keterbacaan:
    1. Konversi ke Grayscale jika masih BGR.
    2. Resize jika dimensi terlalu kecil (standarisasi tinggi minimal ~80px).
    3. Peningkatan kontras adaptif dengan CLAHE (Contrast Limited Adaptive Histogram Equalization).
    4. Gaussian blur ringan untuk mereduksi noise.
    """
    if image is None or image.size == 0:
        return image

    # 1. Konversi ke grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Resize proporsional jika plat terlalu kecil
    h, w = gray.shape[:2]
    if h < 70 or w < 160:
        scale = max(70.0 / max(h, 1), 160.0 / max(w, 1))
        scale = min(scale, 3.0)  # Batasi agar tidak terlalu blur
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # 3. CLAHE untuk penyeimbangan kontras dan pencahayaan tidak merata
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Sedikit gaussian blur untuk meredam artefak kompresi kamera
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return denoised


def extract_plate_text(reader, crop_image: np.ndarray) -> Tuple[str, float, str]:
    """
    Mengekstrak teks dari area plat yang sudah di-crop:
    - Menjalankan pra-pemrosesan
    - Menjalankan EasyOCR dengan allowlist karakter alfanumerik
    - Menghitung confidence skor
    Mengembalikan: (raw_ocr_text, confidence, preprocessed_candidate)
    """
    if crop_image is None or crop_image.size == 0:
        return "", 0.0, ""

    processed = preprocess_plate_image(crop_image)

    # Eksekusi OCR
    results = reader.readtext(
        processed,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -",
    )

    if not results:
        # Coba fallback langsung dengan gambar asli tanpa CLAHE jika hasilnya kosong
        results = reader.readtext(
            crop_image,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -",
        )

    if not results:
        return "", 0.0, ""

    # Gabungkan semua baris deteksi jika plat memiliki beberapa baris (misal tanggal pajak di bawah)
    # Biasanya baris plat utama memiliki ukuran bounding box terbesar / confidence tinggi
    best = max(results, key=lambda item: item[2])
    raw_text = best[1].strip().upper()
    confidence = float(best[2])

    return raw_text, confidence, raw_text

