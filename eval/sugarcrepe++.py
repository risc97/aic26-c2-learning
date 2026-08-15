#!/usr/bin/env python3
"""
Standalone SugarCrepe++ evaluation for PE-Core-L14-336 checkpoint.

Evaluates Image-to-Text (ITT) compositionality: for each sample
(image, P1, P2, N), both positive captions must score higher than
the hard negative caption.

Usage:
    CUDA_VISIBLE_DEVICES="" python3 eval_sugarcrepe.py \
        --checkpoint <path.pt> --coco-dir ./data/coco \
        --scpp-data /tmp/scpp_repo/data
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from open_clip.factory import create_model, get_model_config, get_tokenizer
from open_clip.transform import image_transform_v2, PreprocessCfg

SUBSETS = ['replace_att', 'replace_obj', 'replace_rel', 'swap_att', 'swap_obj']


def parse_args():
    p = argparse.ArgumentParser(description='SugarCrepe++ ITT evaluation')
    p.add_argument('--checkpoint', required=True, help='Path to checkpoint .pt')
    p.add_argument('--coco-dir', default='./data/coco', help='COCO val2017 images dir')
    p.add_argument('--scpp-data', required=True, help='Path to SugarCrepe++ JSON data dir')
    p.add_argument('--model', default='PE-Core-L14-336')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--precision', default='fp32', choices=['fp32', 'bf16'])
    p.add_argument('--output-file', default=None, help='Optional JSON output')
    p.add_argument('--siglip-preprocess', action='store_true', default=False,
                   help='Force SigLIP preprocessing (mean=0.5, std=0.5, squash)')
    p.add_argument('--pretrained-tag', type=str, default='',
                   help='Pretrained tag used during training (e.g. timm_openclip, webli)')
    return p.parse_args()


def load_model(checkpoint_path, model_name, device, precision, pretrained_tag=''):
    print(f"Loading model '{model_name}'...")
    model_cfg = get_model_config(model_name)
    model_kwargs = {}
    if 'init_logit_bias' in model_cfg:
        model_kwargs['init_logit_scale'] = np.log(10)
        model_kwargs['init_logit_bias'] = model_cfg['init_logit_bias']
    model_kwargs['output_tokens'] = True

    model = create_model(model_name, pretrained='', precision=precision,
                         device=device, output_dict=True, **model_kwargs)

    if pretrained_tag:
        from open_clip.pretrained import get_pretrained_cfg as _get_pt_cfg
        pt_cfg = _get_pt_cfg(model_name, pretrained_tag)
        if pt_cfg:
            pp = model.visual.preprocess_cfg
            for k in ['mean', 'std', 'interpolation', 'resize_mode']:
                if k in pt_cfg:
                    pp[k] = pt_cfg[k]

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    if any(k.startswith('module.') for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k[10:]: v for k, v in sd.items()}

    model.load_state_dict(sd, strict=False)
    model.eval()
    model.to(device)
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}")
    return model


def build_preprocess(model, force_siglip=False):
    if force_siglip:
        pp_cfg = PreprocessCfg(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
            size=224, interpolation='bicubic', resize_mode='squash')
    else:
        pp_cfg = model.visual.preprocess_cfg
    if isinstance(pp_cfg, dict):
        size = pp_cfg.get('size', (336, 336))
        if isinstance(size, (tuple, list)):
            size = size[0]
        pp_cfg = PreprocessCfg(
            mean=pp_cfg.get('mean', (0.48145466, 0.4578275, 0.40821073)),
            std=pp_cfg.get('std', (0.26862954, 0.26130258, 0.27577711)),
            size=size,
            interpolation=pp_cfg.get('interpolation', 'bicubic'),
            resize_mode=pp_cfg.get('resize_mode', 'shortest'),
        )
    return image_transform_v2(pp_cfg, is_train=False)


def load_scpp_data(scpp_dir):
    """Load all 5 SugarCrepe++ subsets."""
    all_samples = []
    for subset in SUBSETS:
        path = os.path.join(scpp_dir, f'{subset}.json')
        with open(path) as f:
            samples = json.load(f)
        for s in samples:
            s['subset'] = subset
        all_samples.extend(samples)
        print(f"  {subset}: {len(samples)} samples")
    print(f"  Total: {len(all_samples)} samples")
    return all_samples


def encode_image_batch(model, img_paths, preprocess, device, precision):
    """Encode a batch of images, return L2-normalized features."""
    imgs = []
    valid_indices = []
    for i, path in enumerate(img_paths):
        try:
            img = Image.open(path).convert('RGB')
            imgs.append(preprocess(img))
            valid_indices.append(i)
        except Exception:
            imgs.append(torch.zeros(3, 336, 336))  # dummy
            valid_indices.append(i)

    images = torch.stack(imgs).to(device)
    if precision == 'bf16':
        images = images.to(torch.bfloat16)

    with torch.no_grad():
        result = model.encode_image(images, normalize=True)
        features = result[0] if isinstance(result, tuple) else result
    return features.cpu()


def encode_text_batch(model, texts, tokenizer, device):
    """Encode a batch of texts, return L2-normalized features."""
    tokenized = tokenizer(texts)
    if isinstance(tokenized, dict):
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
    else:
        tokenized = tokenized.to(device)

    with torch.no_grad():
        features = model.encode_text(tokenized, normalize=True)
    return features.cpu()


def evaluate_scpp(model, samples, preprocess, tokenizer, device, args):
    """Run SugarCrepe++ ITT evaluation."""
    images_dir = os.path.join(args.coco_dir, 'val2017')
    batch_size = args.batch_size

    # Collect unique images and index them
    unique_imgs = list(set(s['filename'] for s in samples))
    print(f"\nEncoding {len(unique_imgs)} unique images...")
    img_features = {}  # filename -> feature tensor

    for i in range(0, len(unique_imgs), batch_size):
        batch_fns = unique_imgs[i:i + batch_size]
        batch_paths = [os.path.join(images_dir, fn) for fn in batch_fns]
        feats = encode_image_batch(model, batch_paths, preprocess, device, args.precision)

        for fn, feat in zip(batch_fns, feats):
            img_features[fn] = feat

        if (i // batch_size) % 20 == 0:
            print(f"  Images: {min(i + batch_size, len(unique_imgs))}/{len(unique_imgs)}")

    # Collect all unique captions and index them
    all_captions = []
    caption_index = {}  # (sample_id, pos) -> index in all_captions
    for s in samples:
        for key in ['caption', 'caption2', 'negative_caption']:
            cap_key = (s['id'], s['subset'], key)
            if cap_key not in caption_index:
                caption_index[cap_key] = len(all_captions)
                all_captions.append(s[key])

    print(f"Encoding {len(all_captions)} unique captions...")
    text_features_arr = [None] * len(all_captions)

    for i in range(0, len(all_captions), batch_size):
        batch_texts = all_captions[i:i + batch_size]
        feats = encode_text_batch(model, batch_texts, tokenizer, device)

        for j, feat in enumerate(feats):
            text_features_arr[i + j] = feat

        if (i // batch_size) % 50 == 0:
            print(f"  Texts: {min(i + batch_size, len(all_captions))}/{len(all_captions)}")

    # Evaluate: for each sample, check P1 > N and P2 > N
    print(f"\nComputing ITT accuracy...")
    results_by_subset = {}
    total_correct = 0
    total_samples = 0

    for subset in SUBSETS:
        subset_samples = [s for s in samples if s['subset'] == subset]
        correct = 0

        for s in subset_samples:
            img_feat = img_features[s['filename']]
            p1_feat = text_features_arr[caption_index[(s['id'], subset, 'caption')]]
            p2_feat = text_features_arr[caption_index[(s['id'], subset, 'caption2')]]
            n_feat = text_features_arr[caption_index[(s['id'], subset, 'negative_caption')]]

            sim_p1 = (img_feat @ p1_feat).item()
            sim_p2 = (img_feat @ p2_feat).item()
            sim_n = (img_feat @ n_feat).item()

            if sim_p1 > sim_n and sim_p2 > sim_n:
                correct += 1

        acc = 100.0 * correct / len(subset_samples)
        results_by_subset[subset] = {
            'correct': correct,
            'total': len(subset_samples),
            'accuracy': acc,
        }
        total_correct += correct
        total_samples += len(subset_samples)

    results_by_subset['OVERALL'] = {
        'correct': total_correct,
        'total': total_samples,
        'accuracy': 100.0 * total_correct / total_samples,
    }

    return results_by_subset


def main():
    args = parse_args()

    if torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES', '') == '':
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    device = torch.device(device)
    print(f"Device: {device}")

    model = load_model(args.checkpoint, args.model, device, args.precision, args.pretrained_tag)
    tokenizer = get_tokenizer(args.model)
    preprocess = build_preprocess(model, force_siglip=args.siglip_preprocess)

    samples = load_scpp_data(args.scpp_data)

    t0 = time.time()
    results = evaluate_scpp(model, samples, preprocess, tokenizer, device, args)
    elapsed = time.time() - t0

    print(f"\n{'=' * 55}")
    print("SugarCrepe++ ITT Results")
    print(f"{'=' * 55}")
    print(f"\n{'Subset':<20} {'Correct':>8} {'Total':>8} {'Acc %':>10}")
    print("-" * 46)
    for subset in SUBSETS + ['OVERALL']:
        r = results[subset]
        marker = " <<<" if subset == 'OVERALL' else ""
        print(f"{subset:<20} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>9.2f}%{marker}")

    print(f"\nEvaluation time: {elapsed:.0f}s")

    if args.output_file:
        out = {
            'checkpoint': args.checkpoint,
            'model': args.model,
            'eval_time_s': elapsed,
            'results': results,
        }
        with open(args.output_file, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Saved to {args.output_file}")


if __name__ == '__main__':
    main()
