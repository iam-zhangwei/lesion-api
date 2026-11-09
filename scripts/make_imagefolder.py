"""Build a small ImageFolder dataset (data/train, data/val) from HAM10000-style CSVs.

CSV format: image,MEL,NV,BCC,AKIEC,BKL,DF,VASC  (one-hot, image id without extension)

Example:
  python scripts/make_imagefolder.py --img-dir C:/data/img --train-csv C:/data/train.csv \
      --val-csv C:/data/test.csv --train-per-class 300 --val-per-class 100
"""
import argparse
import csv
import random
import shutil
from pathlib import Path


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        classes = [c for c in reader.fieldnames if c != "image"]
        rows = []
        for r in reader:
            label = max(classes, key=lambda c: float(r[c]))
            rows.append((r["image"], label))
    return rows


def build_split(rows, img_dir, out_dir, per_class, rng):
    by_class = {}
    for name, label in rows:
        by_class.setdefault(label, []).append(name)
    copied = {}
    for label, names in sorted(by_class.items()):
        rng.shuffle(names)
        chosen = names[:per_class] if per_class else names
        dest = out_dir / label
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for name in chosen:
            src = img_dir / name
            if not src.exists():
                src = img_dir / f"{name}.jpg"
            if src.exists():
                shutil.copy2(src, dest / src.name)
                n += 1
        if n == 0 and chosen:
            print(f"WARNING: no images found for class {label}, e.g. {img_dir / chosen[0]}")
        copied[label] = n
    return copied


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--out", default="data")
    p.add_argument("--train-per-class", type=int, default=300, help="0 = all")
    p.add_argument("--val-per-class", type=int, default=100, help="0 = all")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = random.Random(a.seed)
    img_dir, out = Path(a.img_dir), Path(a.out)
    for split, csv_path, n in [
        ("train", a.train_csv, a.train_per_class),
        ("val", a.val_csv, a.val_per_class),
    ]:
        counts = build_split(load_rows(csv_path), img_dir, out / split, n, rng)
        print(split, counts)


if __name__ == "__main__":
    main()
