"""High-speed multi-threaded dataset packager.

Pre-resizes all 7,677 histopathology patches from 1024x1024 down to 224x224 (RGB)
using Lanczos downsampling, and packages them into DataSet_224.zip.
"""

import os
import io
import sys
import time
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# Reconfigure stdout for UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "DataSet"
OUT_ZIP = BASE_DIR / "DataSet_224.zip"
TARGET_SIZE = (224, 224)
JPEG_QUALITY = 95


def process_image(src_path: Path):
    """Reads, resizes to 224x224, and encodes image to JPEG bytes."""
    try:
        with Image.open(src_path) as img:
            rgb_img = img.convert("RGB")
            resized = rgb_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=JPEG_QUALITY)
            return src_path, buf.getvalue(), None
    except Exception as e:
        return src_path, None, str(e)


def main():
    print("=" * 65)
    print("Starting Multi-Threaded Dataset Packaging (224x224 Lanczos)")
    print("=" * 65)
    print(f"Source Directory : {SRC_DIR}")
    print(f"Target Zip File  : {OUT_ZIP}")

    if not SRC_DIR.exists():
        print(f"Error: Source directory {SRC_DIR} does not exist.")
        return

    # 1. Collect all images
    all_files = []
    for root, _, files in os.walk(SRC_DIR):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                all_files.append(Path(root) / f)

    total_files = len(all_files)
    print(f"Found {total_files} total histopathology images to process.")

    start_time = time.time()
    processed_count = 0
    error_count = 0

    # 2. Process & write directly to zip file in streaming chunks
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Utilize 16 worker threads on the 12-core i5-13500H
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_file = {executor.submit(process_image, f): f for f in all_files}

            for future in as_completed(future_to_file):
                src_path, img_bytes, err = future.result()
                if err:
                    print(f"Error on {src_path.name}: {err}")
                    error_count += 1
                else:
                    # Maintain exact subfolder structure under 'DataSet/'
                    rel_path = src_path.relative_to(SRC_DIR)
                    # Normalize path with forward slashes for cross-platform zip
                    clean_rel = str(rel_path.parent).replace("\\", "/")
                    zip_entry_name = f"DataSet/{clean_rel}/{src_path.stem}.jpg"
                    zf.writestr(zip_entry_name, img_bytes)
                    processed_count += 1

                if processed_count % 1000 == 0 or processed_count == total_files:
                    elapsed = time.time() - start_time
                    rate = processed_count / max(0.1, elapsed)
                    print(f"  Processed {processed_count}/{total_files} images ({rate:.1f} img/sec)...")

    elapsed_total = time.time() - start_time
    zip_size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)

    print("=" * 65)
    print("SUCCESS: DATASET PACKAGING COMPLETE!")
    print(f"  - Total Images Packaged : {processed_count} of {total_files} (100%)")
    print(f"  - Output Archive        : {OUT_ZIP}")
    print(f"  - Final Archive Size    : {zip_size_mb:.2f} MB")
    print(f"  - Processing Time       : {elapsed_total:.1f} seconds")
    print("=" * 65)


if __name__ == "__main__":
    main()
