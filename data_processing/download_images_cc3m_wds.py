from PIL import Image
import io
import string
import random
import json
from pathlib import Path
from datasets import load_dataset
from tqdm.auto import tqdm
from argparse import ArgumentParser
import pickle as pkl
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

FILENAME_LENGTH = 8
NUM_SHARDS = 576
SHARD_URL_FMT = "https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/cc3m-train-{i:04d}.tar"


def get_random_string(length=FILENAME_LENGTH):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def save_image(image: Image.Image, image_root_dir: str | Path, random_string: str):
    NUM_SPLITS = 4
    n = len(random_string)
    part_n = n // NUM_SPLITS
    dir_path = Path(image_root_dir)
    for i in range(NUM_SPLITS - 1):
        dir_path = dir_path / random_string[i * part_n:(i + 1) * part_n]
    dir_path.mkdir(exist_ok=True, parents=True)
    file_path = dir_path / f"{random_string}.jpg"
    image.save(str(file_path))
    return str(file_path.relative_to(image_root_dir))


def download_images(image_root_dir: str | Path) -> dict:
    """
    Download CC3M images shard-by-shard using the datasets library's webdataset loader
    (which handles XetHub URL refresh). Checkpoints after every shard for resume.
    """
    image_root_dir = Path(image_root_dir)
    checkpoint_path = image_root_dir / "download_checkpoint.pkl"

    # Resume from checkpoint if available
    metadata_output = {}
    sampled_names = set()
    processed_shards = set()

    if checkpoint_path.exists():
        checkpoint = pkl.load(open(checkpoint_path, "rb"))
        metadata_output = checkpoint.get("metadata", {})
        processed_shards = set(checkpoint.get("processed_shards", set()))
        for v in metadata_output.values():
            sampled_names.add(Path(v["image_path"]).stem)
        print(f"Resuming from checkpoint: {len(metadata_output)} images, {len(processed_shards)} shards done")

    shard_pbar = tqdm(range(NUM_SHARDS), desc="Shards", initial=len(processed_shards))
    consecutive_failures = 0

    for shard_idx in shard_pbar:
        if shard_idx in processed_shards:
            continue

        shard_url = SHARD_URL_FMT.format(i=shard_idx)
        shard_pbar.set_postfix_str(f"{shard_idx:04d} [{len(metadata_output)} imgs]")

        try:
            # Use datasets library for streaming — it handles XetHub URL refresh
            ds = load_dataset(
                "webdataset",
                data_files={"train": [shard_url]},
                split="train",
                streaming=True,
            )

            for row in ds:
                image = row.get("jpg") or row.get("png") or row.get("webp")
                if image is None:
                    continue
                url = row.get("json", {}).get("url", "").strip()
                if not url:
                    continue

                # Generate unique filename
                random_string = get_random_string()
                while random_string in sampled_names:
                    random_string = get_random_string()
                sampled_names.add(random_string)

                image = image.convert("RGB")
                file_path = save_image(image, image_root_dir, random_string)

                metadata_output[len(metadata_output)] = {
                    "image_path": file_path,
                    "url": url,
                }

            processed_shards.add(shard_idx)
            consecutive_failures = 0

        except Exception as e:
            msg = str(e)[:120]
            tqdm.write(f"  shard {shard_idx:04d} FAILED: {msg}  — skipping")
            consecutive_failures += 1
            if consecutive_failures >= 10:
                tqdm.write(f"  {consecutive_failures} consecutive failures, stopping.")
                break
            continue

        # Save checkpoint
        with open(checkpoint_path, "wb") as f:
            pkl.dump({"metadata": metadata_output, "processed_shards": list(processed_shards)}, f)

    shard_pbar.close()
    n = len(processed_shards)
    print(f"\nDone: {len(metadata_output)} images from {n}/{NUM_SHARDS} shards ({n * 100 / NUM_SHARDS:.0f}%)")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return metadata_output


def main():
    parser = ArgumentParser()
    parser.add_argument("--hf-cache-dir", type=str, default="./data_dir/hf_cache")
    parser.add_argument("--image-dir", type=str, default="./data_dir/images")
    parser.add_argument("--output-pkl", type=str, default=None)
    parser.add_argument("--clean-start", action="store_true")

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    image_dir.mkdir(exist_ok=True, parents=True)

    pkl_file = Path(args.output_pkl) if args.output_pkl else (image_dir / "cc3m_images.pkl")
    checkpoint_path = image_dir / "download_checkpoint.pkl"

    if args.clean_start:
        for p in [checkpoint_path]:
            if p.exists():
                p.unlink()
                print(f"Removed {p}")

    # 1. Download images
    downloaded_files = download_images(image_root_dir=image_dir)
    print(f"Downloaded {len(downloaded_files)} images")

    # 2. Load DreamLIP captions
    print("Loading DreamLIP captions (this downloads ~60 GB of CSVs on first run)...")
    dataset = load_dataset(
        "qidouxiong619/dreamlip_long_captions",
        split="train",
        cache_dir=args.hf_cache_dir,
    )
    print(f"DreamLIP loaded: {len(dataset)} samples")

    # 3. Merge by URL
    url_to_path = {}
    for v in downloaded_files.values():
        url_to_path[v["url"]] = v["image_path"]
    print(f"Unique URLs in downloaded images: {len(url_to_path)}")

    image_files = {}
    for row in tqdm(dataset, total=len(dataset), desc="Merging captions"):
        url = row["Image Path"]
        if url not in url_to_path:
            continue
        captions = dict(row)
        captions.pop("Image Path")
        image_files[str(len(image_files))] = {
            "image_path": url_to_path[url],
            "image_url": url,
            "sample_index": len(image_files),
            "captions": captions,
        }

    print("*******************************************************************")
    print(f"Matched: {len(image_files)} image-caption pairs")
    print(f"Unmatched downloaded images (URL not in DreamLIP): {len(downloaded_files) - len(image_files)}")

    with open(pkl_file, "wb") as fp:
        pkl.dump(image_files, fp)
    print(f"Saved to {pkl_file}")


if __name__ == "__main__":
    main()
