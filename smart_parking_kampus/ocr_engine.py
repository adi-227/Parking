import re
import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_plate_image(image: np.ndarray) -> np.ndarray:
    """
    Pra-pemrosesan gambar plat nomor sebelum EasyOCR untuk meningkatkan keterbacaan:
    1. Konversi ke Grayscale jika masih BGR.
    2. Resize jika dimensi terlalu kecil (standarisasi tinggi minimal ~110px).
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

    # 2. Resize proporsional jika plat terlalu kecil agar karakter tajam & terbaca
    h, w = gray.shape[:2]
    if h < 110 or w < 260:
        scale = max(110.0 / max(h, 1), 260.0 / max(w, 1))
        scale = min(scale, 3.5)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # 3. CLAHE untuk penyeimbangan kontras dan pencahayaan tidak merata
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Sedikit gaussian blur untuk meredam artefak kompresi kamera
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 5. Deskewing sudut kemiringan (kamera di tiang samping pos parkir / serong)
    try:
        cur_h, cur_w = denoised.shape[:2]
        edges = cv2.Canny(denoised, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=28, minLineLength=max(25, int(cur_w * 0.15)), maxLineGap=12)
        if lines is not None and len(lines) > 0:
            angles = []
            for l in lines:
                coords = l[0] if len(l.shape) > 1 else l
                x1, y1, x2, y2 = coords
                dx, dy = x2 - x1, y2 - y1
                if abs(dx) > 10:
                    deg = np.degrees(np.arctan2(dy, dx))
                    if abs(deg) <= 30:
                        angles.append(deg)
            if angles:
                median_ang = float(np.median(angles))
                if abs(median_ang) >= 2.0:
                    M = cv2.getRotationMatrix2D((cur_w // 2, cur_h // 2), median_ang, 1.0)
                    denoised = cv2.warpAffine(denoised, M, (cur_w, cur_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass

    return denoised


def extract_plate_text(reader, crop_image: np.ndarray) -> Tuple[str, float, str]:
    """
    Mengekstrak teks dari area plat yang sudah di-crop secara cerdas:
    - Menjalankan pra-pemrosesan kontras & auto-deskewing sudut serong
    - Mengeksekusi EasyOCR
    - Mengabaikan baris tanggal pajak / masa berlaku (misal '09.29') yang sering mendominasi confidence
    - Menggabungkan potongan karakter dari kiri ke kanan jika terpisah beberapa kotak
    - Memilih teks plat nomor kendaraan yang paling lengkap dan sah
    Mengembalikan: (best_cleaned_plate, confidence, raw_ocr_text)
    """
    if crop_image is None or crop_image.size == 0:
        return "", 0.0, ""

    from parking_logic import disambiguate_plate, is_valid_plate_pattern

    processed = preprocess_plate_image(crop_image)

    # 1. Coba baca pada gambar preprocessed (termasuk auto-deskewing kemiringan)
    results = reader.readtext(
        processed,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -.",
    )

    # 2. Cek apakah ada plat valid; jika belum, coba juga gambar asli (raw)
    has_valid = any(is_valid_plate_pattern(disambiguate_plate(r[1])) for r in results) if results else False
    if not has_valid:
        raw_results = reader.readtext(
            crop_image,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -.",
        )
        if raw_results:
            results = list(results) + list(raw_results)

    if not results:
        return "", 0.0, ""

    h_plate = crop_image.shape[0]

    raw_candidates = []
    main_line_boxes = []

    for box, text, conf in results:
        clean_t = text.strip()
        if not clean_t:
            continue

        # Abaikan tanggal pajak di bagian bawah plat (misal 09.29, 09 29, 12.28)
        if re.match(r"^\d{2}[\.\s\-]?\d{2}$", clean_t):
            continue

        cy = sum(pt[1] for pt in box) / 4.0
        cx = sum(pt[0] for pt in box) / 4.0
        bh = max(pt[1] for pt in box) - min(pt[1] for pt in box)

        raw_candidates.append((clean_t, float(conf), bh))

        # Jika di baris atas/tengah plat nomor utama (bukan baris tanggal bawah)
        if cy < (h_plate * 0.72) and conf >= 0.15:
            main_line_boxes.append((cx, clean_t, float(conf), bh))

    # Jika karakter terpecah menjadi beberapa kotak (misal 'B' dan '6991' dan 'JGU')
    # gabungkan dari kiri ke kanan
    if len(main_line_boxes) > 1:
        main_line_boxes.sort(key=lambda item: item[0])
        joined_text = " ".join(item[1] for item in main_line_boxes)
        avg_conf = sum(item[2] for item in main_line_boxes) / len(main_line_boxes)
        max_bh = max(item[3] for item in main_line_boxes)
        raw_candidates.append((joined_text, avg_conf, max_bh))

    if not raw_candidates:
        return "", 0.0, ""

    # Skoring setiap kandidat untuk memilih plat yang paling tepat dan lengkap
    scored_candidates = []
    for text_val, conf_val, bh_val in raw_candidates:
        cleaned = disambiguate_plate(text_val)
        valid_pattern = is_valid_plate_pattern(cleaned)

        score = conf_val * 10.0
        if valid_pattern:
            score += 50.0

        # Bonus untuk struktur lengkap: wilayah + 1-4 digit + huruf seri (misal B 6991 JGU)
        m = re.match(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})$", cleaned)
        if m:
            pfx, num, sfx = m.groups()
            score += len(cleaned) * 4.0  # Plat lengkap lebih baik daripada potongan
            score += len(sfx) * 8.0      # Ada huruf seri di belakang (JGU) sangat diutamakan
            if 1 <= len(num) <= 4:
                score += 15.0

        score += min(bh_val, 50) * 0.5
        scored_candidates.append((score, cleaned, conf_val, text_val))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_cleaned, best_conf, best_raw = scored_candidates[0]

    return best_cleaned, best_conf, best_raw

