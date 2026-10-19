import argparse
import sys
from pathlib import Path
from dataset_validator import validate_dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = ROOT / "dataset" / "data.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "detect"
TARGET_MODEL_PATH = ROOT / "models" / "license_plate_detector.pt"


def train(
    data_yaml: str = str(DEFAULT_DATA_YAML),
    base_model: str = "yolo11n.pt",
    epochs: int = 60,
    imgsz: int = 640,
    batch: int = 8,
    project_dir: str = str(DEFAULT_OUTPUT_DIR),
    experiment_name: str = "plate_detector",
    auto_copy_best: bool = False,
):
    """
    Pipeline training YOLO untuk plat nomor:
    1. Validasi dataset terlebih dahulu (tidak jalan jika kosong / corrupt).
    2. Konfigurasi model & hyperparameter yang aman di Windows.
    3. Eksekusi training Ultralytics.
    4. Evaluasi metrik dan opsi copy bobot terbaik.
    """
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        print(f"[ERROR] File konfigurasi dataset tidak ditemukan di: {yaml_path}")
        print("Silakan buat atau sesuaikan dataset/data.yaml terlebih dahulu.")
        sys.exit(1)

    dataset_root = yaml_path.parent
    print("=" * 60)
    print("MEMERIKSA DAN MEMVALIDASI DATASET SEBELUM TRAINING...")
    print(f"Direktori dataset: {dataset_root}")
    print(f"File data.yaml   : {yaml_path}")
    print("=" * 60)

    report = validate_dataset(str(dataset_root), yaml_path=str(yaml_path))

    if report["warnings"]:
        print("\n[PERINGATAN]:")
        for w in report["warnings"]:
            print(f" - {w}")

    if report["errors"]:
        print("\n[ERROR VALIDASI DATASET]:")
        for err in report["errors"]:
            print(f" - {err}")
        print("\nTraining DIBATALKAN karena dataset mengandung error.")
        sys.exit(1)

    total_images = report["train_images_count"] + report["val_images_count"]
    if total_images == 0:
        print("\n[PERINGATAN PENTING]")
        print("Dataset masih KOSONG (0 gambar train, 0 gambar val).")
        print("Sesuai instruksi, training TIDAK dijalankan dengan dataset kosong.")
        print("Masukkan dataset gambar & label ke dalam folder dataset/ sebelum memulai training.")
        sys.exit(0)

    print("\nDataset VALID. Rincian:")
    print(f" - Gambar Train: {report['train_images_count']}, Label: {report['train_labels_count']}")
    print(f" - Gambar Val  : {report['val_images_count']}, Label: {report['val_labels_count']}")
    print(f" - Kelas ditemukan: {report['class_ids_found']}")

    # Import ultralytics hanya saat dataset siap dijalankan
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Paket 'ultralytics' belum terpasang. Jalankan: pip install ultralytics")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"MEMULAI TRAINING YOLO (Base Model: {base_model}, Epochs: {epochs}, Imgsz: {imgsz}, Batch: {batch})")
    print("=" * 60)

    try:
        model = YOLO(base_model)
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(project_dir),
            name=experiment_name,
            exist_ok=True,
            verbose=True,
        )

        best_weights = Path(project_dir) / experiment_name / "weights" / "best.pt"
        if best_weights.exists():
            print("\n" + "=" * 60)
            print("[BERHASIL] Training selesai!")
            print(f"Bobot terbaik tersimpan di: {best_weights}")
            print("=" * 60)
            
            if auto_copy_best:
                import shutil
                TARGET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_weights, TARGET_MODEL_PATH)
                print(f"Model berhasil disalin ke: {TARGET_MODEL_PATH}")
            else:
                print(f"Untuk mengaktifkan di aplikasi web, salin file tersebut ke:")
                print(f"  {TARGET_MODEL_PATH}")
        else:
            print("[INFO] Training selesai, periksa folder runs untuk hasil bobot.")

    except Exception as exc:
        print(f"\n[ERROR SAAT TRAINING]: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script Training YOLO Plat Nomor Kampus")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML), help="Path ke data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Bobot dasar YOLO (misal yolo11n.pt / yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=60, help="Jumlah epoch")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran resolusi gambar input")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--project", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Direktori output")
    parser.add_argument("--name", type=str, default="plate_detector", help="Nama eksperimen")
    parser.add_argument("--auto-copy", action="store_true", help="Salin otomatis best.pt ke models/license_plate_detector.pt")
    parser.add_argument("--validate-only", action="store_true", help="Hanya jalankan validasi dataset tanpa training")

    args = parser.parse_args()

    if args.validate_only:
        p = Path(args.data).resolve()
        rep = validate_dataset(str(p.parent), yaml_path=str(p))
        print("Laporan Validasi:", rep)
    else:
        train(
            data_yaml=args.data,
            base_model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project_dir=args.project,
            experiment_name=args.name,
            auto_copy_best=args.auto_copy,
        )
