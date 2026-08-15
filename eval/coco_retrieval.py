#!/usr/bin/env python3
"""
Standalone MS-COCO Recall@k evaluation for PE-Core-L14-336 checkpoint.

Runs on CPU by default to avoid disrupting active GPU training.

Usage:
    CUDA_VISIBLE_DEVICES="" python eval_coco_retrieval.py \
        --checkpoint checkpoints/PE-Core-L14-336-0508-0854/checkpoints/epoch_2.pt \
        --coco-dir ./data/coco --split val2017
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from open_clip.factory import create_model, get_model_config, get_tokenizer
from open_clip.transform import image_transform_v2, PreprocessCfg

try:
    from pycocotools.coco import COCO
except ImportError:
    print("Error: pycocotools not installed. Run: pip install pycocotools")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="MS-COCO Recall@k evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--coco-dir", type=str, default="./data/coco",
                        help="Directory containing COCO data (images + annotations)")
    parser.add_argument("--split", type=str, default="val2017",
                        choices=["val2017", "val2014"],
                        help="COCO split to evaluate on")
    parser.add_argument("--model", type=str, default="PE-Core-L14-336",
                        help="Model architecture name")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for feature extraction (cpu)")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "bf16"],
                        help="Precision for inference")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Optional JSON file to save metrics")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Limit number of images (for quick testing)")
    parser.add_argument("--pretrained-tag", type=str, default='',
                        help="Pretrained tag used during training (e.g. timm_openclip, webli). "
                             "Looked up in pretrained.py to get correct mean/std/resize.")
    parser.add_argument("--siglip-preprocess", action="store_true", default=False,
                        help="Force SigLIP preprocessing (mean=0.5, std=0.5, squash) "
                             "- fallback if pretrained-tag not available")
    return parser.parse_args()


def load_model(checkpoint_path, model_name, device, precision, pretrained_tag=''):
    """Create model architecture and load checkpoint weights."""
    print(f"Loading model '{model_name}'...")

    model_cfg = get_model_config(model_name)
    if model_cfg is None:
        raise RuntimeError(f"Model config for '{model_name}' not found.")

    # SigLIP-specific model kwargs
    model_kwargs = {}
    if 'init_logit_bias' in model_cfg:
        model_kwargs['init_logit_scale'] = np.log(10)
        model_kwargs['init_logit_bias'] = model_cfg['init_logit_bias']

    # Enable output_tokens (model was trained with this)
    model_kwargs['output_tokens'] = True

    model = create_model(
        model_name,
        pretrained='',  # Don't load pretrained; we load from checkpoint
        precision=precision,
        device=device,
        output_dict=True,
        **model_kwargs,
    )

    # Apply pretrained cfg preprocessing (mean/std/resize) without downloading weights
    if pretrained_tag:
        from open_clip.pretrained import get_pretrained_cfg as _get_pt_cfg
        pt_cfg = _get_pt_cfg(model_name, pretrained_tag)
        if pt_cfg:
            pp = model.visual.preprocess_cfg
            for k in ['mean', 'std', 'interpolation', 'resize_mode']:
                if k in pt_cfg:
                    pp[k] = pt_cfg[k]
            print(f"  Applied pretrained '{pretrained_tag}' preprocessing: "
                  f"mean={pp['mean']}, std={pp['std']}, "
                  f"interpolation={pp.get('interpolation')}, "
                  f"resize_mode={pp.get('resize_mode')}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get('state_dict', checkpoint)
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    # Strip DDP / torch.compile prefixes
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k[10:]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {[k for k in missing[:5]]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {[k for k in unexpected[:5]]}...")

    model.eval()
    model.to(device)
    print("  Model loaded successfully.")
    return model


def build_preprocess(model, force_siglip=False):
    """Build image preprocessing pipeline."""
    if force_siglip:
        pp_cfg = PreprocessCfg(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            size=224,
            interpolation='bicubic',
            resize_mode='squash',
        )
    else:
        pp_cfg = model.visual.preprocess_cfg
    # pp_cfg might be a dict or PreprocessCfg object
    if isinstance(pp_cfg, dict):
        size = pp_cfg.get('size', (224, 224))
        if isinstance(size, (tuple, list)):
            size = size[0] if len(size) > 1 else 224
        pp_cfg = PreprocessCfg(
            mean=pp_cfg.get('mean', (0.48145466, 0.4578275, 0.40821073)),
            std=pp_cfg.get('std', (0.26862954, 0.26130258, 0.27577711)),
            size=size,
            interpolation=pp_cfg.get('interpolation', 'bicubic'),
            resize_mode=pp_cfg.get('resize_mode', 'shortest'),
        )
    preprocess = image_transform_v2(pp_cfg, is_train=False)
    print(f"  Preprocessing: mean={pp_cfg.mean}, std={pp_cfg.std}, size={pp_cfg.size}")
    return preprocess


def load_coco_data(coco_dir, split, num_images=None):
    """Load COCO dataset images and captions."""
    ann_file = os.path.join(coco_dir, 'annotations', f'captions_{split}.json')
    if not os.path.exists(ann_file):
        raise FileNotFoundError(
            f"Captions file not found at {ann_file}. "
            f"First run: cd {coco_dir} && unzip annotations_trainval2017.zip"
        )

    print(f"Loading COCO annotations from {ann_file}...")
    coco = COCO(ann_file)

    img_ids = sorted(coco.getImgIds())
    if num_images:
        img_ids = img_ids[:num_images]

    images_info = coco.loadImgs(img_ids)
    id_to_filename = {img['id']: img['file_name'] for img in images_info}
    id_to_captions = {}
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        id_to_captions[img_id] = [ann['caption'] for ann in anns]

    total_captions = sum(len(c) for c in id_to_captions.values())
    print(f"  Loaded {len(img_ids)} images, {total_captions} captions")
    return img_ids, id_to_filename, id_to_captions


def compute_features(model, img_ids, id_to_filename, id_to_captions,
                     preprocess, tokenizer, device, args):
    """Encode all images and captions, returning L2-normalized feature matrices."""
    images_dir = os.path.join(args.coco_dir, args.split)

    # --- Pre-scan: build img_id -> index mapping ---
    image_id_to_idx = {}
    valid_img_ids = []
    for img_id in img_ids:
        img_path = os.path.join(images_dir, id_to_filename[img_id])
        if os.path.exists(img_path):
            image_id_to_idx[img_id] = len(valid_img_ids)
            valid_img_ids.append(img_id)

    num_valid = len(valid_img_ids)
    if num_valid < len(img_ids):
        print(f"  Warning: {len(img_ids) - num_valid} images not found on disk")

    # --- Encode images ---
    print(f"\nEncoding {num_valid} images...")
    all_image_features = []
    batch_size = args.batch_size

    with torch.no_grad():
        for i in range(0, num_valid, batch_size):
            batch_ids = valid_img_ids[i:i + batch_size]
            batch_imgs = []
            for img_id in batch_ids:
                img_path = os.path.join(images_dir, id_to_filename[img_id])
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_imgs.append(preprocess(img))
                except Exception as e:
                    print(f"  Warning: failed {img_path}: {e}")
                    batch_imgs.append(torch.zeros(3, preprocess.transforms[0].size,
                                                  preprocess.transforms[0].size))

            images = torch.stack(batch_imgs).to(device)
            if args.precision == 'bf16':
                images = images.to(torch.bfloat16)

            # Use encode_image (returns (features, tokens) when output_tokens=True)
            result = model.encode_image(images, normalize=True)
            features = result[0] if isinstance(result, tuple) else result
            all_image_features.append(features.cpu())

            if (i // batch_size) % 50 == 0:
                print(f"  Images: {min(i + batch_size, num_valid)}/{num_valid}")

    all_image_features = torch.cat(all_image_features, dim=0)
    print(f"  Image features: {all_image_features.shape}")

    # --- Build text list ---
    all_texts = []
    text_img_mapping = []  # text_idx -> image_feature_idx
    for img_id in valid_img_ids:
        for caption in id_to_captions[img_id]:
            all_texts.append(caption)
            text_img_mapping.append(image_id_to_idx[img_id])

    # --- Encode texts ---
    num_texts = len(all_texts)
    print(f"Encoding {num_texts} captions...")
    all_text_features = []

    with torch.no_grad():
        for i in range(0, num_texts, batch_size):
            batch_texts = all_texts[i:i + batch_size]
            tokenized = tokenizer(batch_texts)
            if isinstance(tokenized, dict):
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
            else:
                tokenized = tokenized.to(device)

            # encode_text returns features (not tuple) for output_tokens=False
            features = model.encode_text(tokenized, normalize=True)
            all_text_features.append(features.cpu())

            if (i // batch_size) % 100 == 0:
                print(f"  Texts: {min(i + batch_size, num_texts)}/{num_texts}")

    all_text_features = torch.cat(all_text_features, dim=0)
    print(f"  Text features: {all_text_features.shape}")

    return all_image_features, all_text_features, text_img_mapping


def compute_recall_metrics(image_features, text_features, text_img_mapping):
    """Compute T2I and I2T Recall@k metrics for MS-COCO.

    MS-COCO has 5 captions per image:
      - T2I: each text is a query → find its paired image (5K images, 25K texts)
      - I2T: each image is a query → find its 5 correct captions among 25K
    """
    metrics = {}

    # similarity: [num_texts, num_images]
    print("Computing similarity matrix...")
    sim = text_features @ image_features.t()
    num_images = sim.shape[1]

    # --- T2I ---
    print("Computing T2I metrics...")
    ground_truth = torch.tensor(text_img_mapping, dtype=torch.long)
    ranking = torch.argsort(sim, dim=1, descending=True)
    preds = (ranking == ground_truth.unsqueeze(1)).nonzero(as_tuple=True)[1].numpy()

    for k in [1, 5, 10]:
        metrics[f"text_to_image_R@{k}"] = float(np.mean(preds < k) * 100)
    metrics["text_to_image_mean_rank"] = float(preds.mean() + 1)
    metrics["text_to_image_median_rank"] = float(np.median(preds) + 1)

    # --- I2T ---
    print("Computing I2T metrics...")
    sim_it = sim.t()  # [num_images, num_texts]
    correct = torch.zeros(num_images, sim_it.shape[1], dtype=torch.bool)
    for t_idx, i_idx in enumerate(text_img_mapping):
        correct[i_idx, t_idx] = True

    ranking_it = torch.argsort(sim_it, dim=1, descending=True)

    for k in [1, 5, 10]:
        top_k = ranking_it[:, :k]
        hits = torch.gather(correct, 1, top_k).any(dim=1)
        metrics[f"image_to_text_R@{k}"] = float(hits.float().mean() * 100)

    # Mean rank of first correct text per image
    first_ranks = []
    for i in range(num_images):
        correct_idxs = correct[i].nonzero(as_tuple=True)[0]
        if len(correct_idxs):
            ranks = (ranking_it[i].unsqueeze(1) == correct_idxs.unsqueeze(0)).nonzero(as_tuple=True)[0] + 1
            first_ranks.append(ranks.min().item())
    first_ranks = np.array(first_ranks)
    metrics["image_to_text_mean_rank"] = float(first_ranks.mean())
    metrics["image_to_text_median_rank"] = float(np.median(first_ranks))

    return metrics


def main():
    args = parse_args()

    # Use CPU when CUDA_VISIBLE_DEVICES is empty (don't disrupt training)
    if torch.cuda.is_available():
        cuvis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuvis == '':
            device = 'cpu'
            print("CUDA_VISIBLE_DEVICES is empty — using CPU to avoid disrupting training.")
        else:
            device = f'cuda:{cuvis.split(",")[0]}'
            print(f"WARNING: Using GPU {device}. This may disrupt training!")
    else:
        device = 'cpu'

    device = torch.device(device)
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, args.model, device, args.precision, args.pretrained_tag)
    tokenizer = get_tokenizer(args.model)
    preprocess = build_preprocess(model, force_siglip=args.siglip_preprocess)

    # Load COCO data
    img_ids, id_to_filename, id_to_captions = load_coco_data(
        args.coco_dir, args.split, num_images=args.num_images
    )

    # Encode
    t0 = time.time()
    image_features, text_features, text_img_mapping = compute_features(
        model, img_ids, id_to_filename, id_to_captions,
        preprocess, tokenizer, device, args
    )
    t_feat = time.time() - t0
    print(f"\nFeature extraction: {t_feat:.0f}s")

    # Compute metrics
    t0 = time.time()
    metrics = compute_recall_metrics(image_features, text_features, text_img_mapping)
    t_metrics = time.time() - t0
    print(f"Metric computation: {t_metrics:.0f}s")

    # Print
    print("\n" + "=" * 55)
    print("MS-COCO Recall@k Results")
    print("=" * 55)
    print(f"\nText-to-Image Retrieval ({len(text_features)} queries → {len(image_features)} images):")
    for k in [1, 5, 10]:
        print(f"  R@{k}: {metrics[f'text_to_image_R@{k}']:.2f}%")
    print(f"  Mean Rank: {metrics['text_to_image_mean_rank']:.2f}")
    print(f"  Median Rank: {metrics['text_to_image_median_rank']:.2f}")

    print(f"\nImage-to-Text Retrieval ({len(image_features)} queries → {len(text_features)} texts):")
    for k in [1, 5, 10]:
        print(f"  R@{k}: {metrics[f'image_to_text_R@{k}']:.2f}%")
    print(f"  Mean Rank: {metrics['image_to_text_mean_rank']:.2f}")
    print(f"  Median Rank: {metrics['image_to_text_median_rank']:.2f}")
    print()

    if args.output_file:
        out = {
            'checkpoint': args.checkpoint,
            'model': args.model,
            'split': args.split,
            'num_images': len(image_features),
            'num_texts': len(text_features),
            'feature_time_s': t_feat,
            'metric_time_s': t_metrics,
            **metrics,
        }
        with open(args.output_file, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Saved to {args.output_file}")

    return metrics


if __name__ == "__main__":
    main()
