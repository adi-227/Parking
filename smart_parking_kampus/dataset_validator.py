import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_yolo_annotation_line(line: str) -> Tuple[bool, Optional[str], Optional[int], Optional[List[float]]]:
    """
    Validasi satu baris anotasi format YOLO:
    Format: <class_id> <x_center> <y_center> <width> <height>
    Semua koordinat ternormalisasi antara 0.0 dan 1.0.
    """
    parts = line.strip().split()
    if not parts:
        return True, None, None, None  # Baris kosong diizinkan diabaikan
    if len(parts) != 5:
        return False, f"Format tidak valid: diharapkan 5 elemen, didapat {len(parts)}", None, None

    try:
        class_id = int(parts[0])
    except ValueError:
        return False, f"Class ID bukan integer: '{parts[0]}'", None, None

    try:
        coords = [float(p) for p in parts[1:]]
    except ValueError:
        return False, f"Koordinat bukan float valid: {parts[1:]}", class_id, None

    x, y, w, h = coords
    for val, name in zip(coords, ["x_center", "y_center", "width", "height"]):
        if val < 0.0 or val > 1.0:
            return False, f"Koordinat {name} ({val}) di luar batas rentang [0.0, 1.0]", class_id, coords

    if w <= 0.0 or h <= 0.0:
        return False, f"Lebar atau tinggi box bernilai <= 0: w={w}, h={h}", class_id, coords

    return True, None, class_id, coords


def validate_dataset(dataset_dir: str, yaml_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Validasi menyeluruh struktur dan isi dataset YOLO.
    Memeriksa:
    - Keberadaan data.yaml
    - Struktur direktori images/train, images/val, labels/train, labels/val
    - Kesesuaian pasangan file gambar dan label .txt
    - Format anotasi YOLO, ID kelas, batas koordinat [0, 1]
    - File rusak/kosong
    - Data leakage antara split train dan val
    """
    dataset_path = Path(dataset_dir).resolve()
    report: Dict[str, Any] = {
        "is_valid": False,
        "dataset_dir": str(dataset_path),
        "yaml_found": False,
        "yaml_content": None,
        "errors": [],
        "warnings": [],
        "train_images_count": 0,
        "train_labels_count": 0,
        "val_images_count": 0,
        "val_labels_count": 0,
        "class_ids_found": set(),
        "leakage_count": 0,
    }

    yaml_file = Path(yaml_path).resolve() if yaml_path else dataset_path / "data.yaml"
    if yaml_file.exists():
        report["yaml_found"] = True
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                report["yaml_content"] = yaml.safe_load(f)
        except Exception as e:
            report["errors"].append(f"Gagal membaca YAML ({yaml_file.name}): {str(e)}")
    else:
        report["warnings"].append(f"File configuration {yaml_file.name} tidak ditemukan.")

    # Cek folder
    img_train_dir = dataset_path / "images" / "train"
    img_val_dir = dataset_path / "images" / "val"
    lbl_train_dir = dataset_path / "labels" / "train"
    lbl_val_dir = dataset_path / "labels" / "val"

    for d, name in [
        (img_train_dir, "images/train"),
        (img_val_dir, "images/val"),
        (lbl_train_dir, "labels/train"),
        (lbl_val_dir, "labels/val"),
    ]:
        if not d.exists() or not d.is_dir():
            report["warnings"].append(f"Direktori {name} belum ada atau bukan folder.")

    train_imgs = {f.stem: f for f in img_train_dir.glob("*") if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS} if img_train_dir.exists() else {}
    val_imgs = {f.stem: f for f in img_val_dir.glob("*") if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS} if img_val_dir.exists() else {}
    train_lbls = {f.stem: f for f in lbl_train_dir.glob("*.txt")} if lbl_train_dir.exists() else {}
    val_lbls = {f.stem: f for f in lbl_val_dir.glob("*.txt")} if lbl_val_dir.exists() else {}

    report["train_images_count"] = len(train_imgs)
    report["train_labels_count"] = len(train_lbls)
    report["val_images_count"] = len(val_imgs)
    report["val_labels_count"] = len(val_lbls)

    # Deteksi dataset kosong
    total_imgs = len(train_imgs) + len(val_imgs)
    if total_imgs == 0:
        report["warnings"].append("Dataset belum berisi file gambar (dataset kosong). Training belum dapat dijalankan.")
        report["class_ids_found"] = list(report["class_ids_found"])
        return report

    # Cek kebocoran data (Data Leakage antara train dan val)
    common_stems = set(train_imgs.keys()).intersection(set(val_imgs.keys()))
    if common_stems:
        report["leakage_count"] = len(common_stems)
        report["errors"].append(f"Terdeteksi Data Leakage: {len(common_stems)} nama file gambar ada di 'train' dan 'val' sekaligus: {list(common_stems)[:5]}")

    # Validasi kesesuaian gambar dan label pada Train
    unpaired_train_imgs = set(train_imgs.keys()) - set(train_lbls.keys())
    if unpaired_train_imgs:
        report["warnings"].append(f"{len(unpaired_train_imgs)} gambar di train tidak memiliki file label .txt pendamping (contoh: {list(unpaired_train_imgs)[:3]})")

    unpaired_train_lbls = set(train_lbls.keys()) - set(train_imgs.keys())
    if unpaired_train_lbls:
        report["warnings"].append(f"{len(unpaired_train_lbls)} label di train tidak memiliki file gambar pendamping (contoh: {list(unpaired_train_lbls)[:3]})")

    # Validasi kesesuaian gambar dan label pada Val
    unpaired_val_imgs = set(val_imgs.keys()) - set(val_lbls.keys())
    if unpaired_val_imgs:
        report["warnings"].append(f"{len(unpaired_val_imgs)} gambar di val tidak memiliki file label .txt pendamping (contoh: {list(unpaired_val_imgs)[:3]})")

    unpaired_val_lbls = set(val_lbls.keys()) - set(val_imgs.keys())
    if unpaired_val_lbls:
        report["warnings"].append(f"{len(unpaired_val_lbls)} label di val tidak memiliki file gambar pendamping (contoh: {list(unpaired_val_lbls)[:3]})")

    # Validasi isi file label .txt
    def check_labels(lbl_dict: Dict[str, Path], split_name: str):
        for stem, path in lbl_dict.items():
            try:
                content = path.read_text(encoding="utf-8").strip()
            except Exception as e:
                report["errors"].append(f"Gagal membaca label {split_name}/{path.name}: {str(e)}")
                continue

            if not content:
                report["warnings"].append(f"File label kosong: {split_name}/{path.name}")
                continue

            for idx, line in enumerate(content.splitlines(), start=1):
                if not line.strip():
                    continue
                valid, err, cid, _ = validate_yolo_annotation_line(line)
                if not valid:
                    report["errors"].append(f"{split_name}/{path.name} (baris {idx}): {err}")
                elif cid is not None:
                    report["class_ids_found"].add(cid)

    check_labels(train_lbls, "train")
    check_labels(val_lbls, "val")

    report["class_ids_found"] = sorted(list(report["class_ids_found"]))
    
    # Validasi class ID: untuk deteksi plat tunggal seharusnya hanya class 0
    if report["class_ids_found"] and report["class_ids_found"] != [0]:
        report["warnings"].append(f"ID kelas yang ditemukan adalah {report['class_ids_found']}. Pastikan hanya kelas 0 yang digunakan jika mendeteksi satu kelas plat.")

    report["is_valid"] = len(report["errors"]) == 0 and total_imgs > 0
    return report

