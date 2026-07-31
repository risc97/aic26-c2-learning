"""Resize all CC3M images so the shorter side is at most TARGET_SIZE pixels.

This preserves the training pipeline's needs (RandomResizedCrop(224) from a 256px
source is identical to operating on the original) while dramatically cutting disk
usage by stripping high-res pixels and EXIF bloat.
"""

from PIL import Image
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from argparse import ArgumentParser

TARGET_SIZE = 256  # shorter side — training crop is 224, so 256 gives enough slack
JPEG_QUALITY = 85


def resize_one(path: Path) -> tuple[str, float]:
    """Resize a single image in-place. Returns (path_str, compression_ratio)."""
    try:
        original_size = path.stat().st_size
        img = Image.open(path).convert("RGB")
        w, h = img.size
        shorter = min(w, h)

        if shorter <= TARGET_SIZE:
            # Already small enough — still strip metadata to save space
            img.save(path, "JPEG", quality=JPEG_QUALITY, optimize=True)
            new_size = path.stat().st_size
            return (str(path), new_size / max(original_size, 1))

        scale = TARGET_SIZE / shorter
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img.save(path, "JPEG", quality=JPEG_QUALITY, optimize=True)
        new_size = path.stat().st_size
        return (str(path), new_size / max(original_size, 1))

    except Exception as e:
        return (str(path), -1.0)  # sentinel for errors


def main():
    parser = ArgumentParser(description="Batch-resize images to save disk space")
    parser.add_argument("--image-dir", type=str, default="/workspace/data/images")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-size", type=int, default=256,
                        help="Max size for the shorter image side (default: 256)")
    parser.add_argument("--quality", type=int, default=85,
                        help="JPEG quality 1–100 (default: 85)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    paths = list(image_dir.rglob("*.jpg"))
    print(f"Found {len(paths)} images in {image_dir}")

    if not paths:
        print("No images found.")
        return

    resized = skipped = errors = 0
    total_saved = 0

    with Pool(args.workers) as pool:
        for path_str, ratio in tqdm(
            pool.imap_unordered(resize_one, paths),
            total=len(paths),
            desc="Resizing",
        ):
            if ratio < 0:
                errors += 1
            elif ratio >= 1.0:
                skipped += 1  # already small enough, no shrink
            else:
                resized += 1
                # Estimate bytes saved (rough)
                p = Path(path_str)
                total_saved += p.stat().st_size * (1 - ratio) / ratio

    print(f"Resized:  {resized}")
    print(f"Skipped:  {skipped} (already ≤{args.target_size}px)")
    print(f"Errors:   {errors}")
    print(f"Estimated space saved: {total_saved / (1024**3):.1f} GB")


if __name__ == "__main__":
    main()
