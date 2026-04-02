import argparse
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def preprocess_dataset(src_root: Path, dst_root: Path, img_size: int = 64):
    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset folder not found: {src_root}")

    image_count = 0
    skipped_count = 0

    for class_dir in sorted([p for p in src_root.iterdir() if p.is_dir()]):
        out_class_dir = dst_root / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for src_file in sorted(class_dir.rglob("*")):
            if not src_file.is_file():
                continue

            if src_file.suffix.lower() not in VALID_EXTS:
                skipped_count += 1
                continue

            rel_path = src_file.relative_to(class_dir)
            out_file = out_class_dir / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                with Image.open(src_file) as img:
                    # Match your default ops: grayscale + resize
                    img = img.convert("L")
                    img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)

                    # Keep original extension if possible; fallback to PNG when needed.
                    save_ext = out_file.suffix.lower()
                    if save_ext in {".jpg", ".jpeg"}:
                        img.save(out_file, quality=95)
                    elif save_ext in {".png", ".bmp", ".tif", ".tiff", ".webp"}:
                        img.save(out_file)
                    else:
                        out_file = out_file.with_suffix(".png")
                        img.save(out_file)

                    image_count += 1

            except (UnidentifiedImageError, OSError):
                skipped_count += 1

    print(f"Done. Processed images: {image_count}")
    print(f"Skipped files: {skipped_count}")
    print(f"Output dataset: {dst_root}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a preprocessed copy of an ImageFolder dataset (grayscale + resize)."
    )
    parser.add_argument("--src", required=True, help="Source dataset root")
    parser.add_argument("--dst", required=True, help="Destination dataset root")
    parser.add_argument("--img-size", type=int, default=64, help="Target image size (default: 64)")
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    preprocess_dataset(src_root, dst_root, img_size=args.img_size)

if __name__ == "__main__":
    main()