#!/usr/bin/env python3
"""CountBench evaluation — 9-way counting classification for VLMs."""
import json, os, sys, time, torch, numpy as np
from collections import Counter
sys.path.insert(0, 'src')
from datasets import load_dataset
from open_clip.factory import create_model, get_model_config, get_tokenizer
from open_clip.transform import image_transform_v2, PreprocessCfg
from open_clip.pretrained import get_pretrained_cfg

NUM_WORDS = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
NUM_MAP = {w: i+2 for i, w in enumerate(NUM_WORDS)}  # 'two' -> 2, etc.

def load_model(ckpt_path, model_name, pretrained_tag, device):
    cfg = get_model_config(model_name)
    kw = {}
    if 'init_logit_bias' in cfg:
        kw['init_logit_scale'] = np.log(10)
        kw['init_logit_bias'] = cfg['init_logit_bias']
    kw['output_tokens'] = True

    model = create_model(model_name, pretrained='', precision='fp32', device=device,
                         output_dict=True, **kw)

    if pretrained_tag:
        pt_cfg = get_pretrained_cfg(model_name, pretrained_tag)
        if pt_cfg:
            pp = model.visual.preprocess_cfg
            for k in ['mean', 'std', 'interpolation', 'resize_mode']:
                if k in pt_cfg: pp[k] = pt_cfg[k]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    if any(k.startswith('module.') for k in sd): sd = {k[7:]: v for k, v in sd.items()}
    if any(k.startswith('_orig_mod.') for k in sd): sd = {k[10:]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

def build_pp(model):
    pp_cfg = model.visual.preprocess_cfg
    s = pp_cfg.get('size', (224,224))
    s = s[0] if isinstance(s, (tuple,list)) else s
    return image_transform_v2(PreprocessCfg(
        mean=pp_cfg['mean'], std=pp_cfg['std'], size=s,
        interpolation=pp_cfg.get('interpolation','bicubic'),
        resize_mode=pp_cfg.get('resize_mode','shortest')), is_train=False)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() and
                          os.environ.get('CUDA_VISIBLE_DEVICES','') != '' else 'cpu')
    print(f'Device: {device}')

    ds = load_dataset('nielsr/countbench', split='train')
    print(f'CountBench: {len(ds)} images, numbers: 2-10 ({60} each)')

    configs = [
        ('PE-Core-L14-336', 'checkpoints/PE-Core-L14-336-0808-0444/checkpoints/epoch_1.pt',
         'timm_openclip', 'checkpoints/PE-Core-L14-336-0808-0444/countbench_results.json'),
        ('ViT-B-16-SigLIP', 'checkpoints/reference/c2lip_npc_xac.pt',
         'webli', 'checkpoints/reference/countbench_results.json'),
    ]

    for model_name, ckpt_path, pt_tag, out_path in configs:
        print(f'\n{"="*50}')
        print(f'{model_name} (pretrained={pt_tag})')
        print('='*50)

        model = load_model(ckpt_path, model_name, pt_tag, device)
        tokenizer = get_tokenizer(model_name)
        pp = build_pp(model)
        pp_cfg = model.visual.preprocess_cfg
        print(f'PP: mean={pp_cfg["mean"]}, std={pp_cfg["std"]}, '
              f'interp={pp_cfg.get("interpolation")}, resize={pp_cfg.get("resize_mode")}')

        # Encode 9 candidate captions once
        num_tokens = tokenizer(NUM_WORDS)
        if isinstance(num_tokens, dict):
            num_tokens = {k: v.to(device) for k, v in num_tokens.items()}
        else:
            num_tokens = num_tokens.to(device)
        with torch.no_grad():
            num_feats = model.encode_text(num_tokens, normalize=True).cpu()
        # num_feats: [9, dim]

        # Evaluate each image
        t0 = time.time()
        preds, gts = [], []
        skipped = 0
        for i, sample in enumerate(ds):
            if sample['image'] is None:
                skipped += 1
                continue
            img = sample['image'].convert('RGB')
            img_tensor = pp(img).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img_tensor, normalize=True)
                if isinstance(img_feat, tuple):
                    img_feat = img_feat[0]
                img_feat = img_feat.cpu()

            sims = (img_feat @ num_feats.t()).squeeze(0)  # [9]
            pred = sims.argmax().item() + 2  # 0-index -> 2..10
            gt = sample['number']
            preds.append(pred)
            gts.append(gt)
        if skipped:
            print(f'  Skipped {skipped} images with dead URLs')

        # Metrics
        preds = np.array(preds); gts = np.array(gts)
        acc = (preds == gts).mean() * 100
        mae = np.abs(preds - gts).mean()
        elapsed = time.time() - t0

        # Per-number accuracy
        per_num = {}
        for n in range(2, 11):
            mask = gts == n
            per_num[str(n)] = float((preds[mask] == n).mean() * 100)

        results = {'checkpoint': ckpt_path, 'model': model_name,
                   'accuracy': acc, 'mae': mae, 'eval_time_s': elapsed,
                   'per_number_accuracy': per_num,
                   'pred_histogram': dict(sorted(Counter(preds.tolist()).items()))}
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f'  Accuracy: {acc:.1f}%  |  MAE: {mae:.3f}  |  Time: {elapsed:.0f}s')
        print(f'  Per-number: ' + ' '.join(f'{n}:{per_num[str(n)]:.0f}%' for n in range(2,11)))
        print(f'  Pred distribution: {results["pred_histogram"]}')
        print(f'  Saved to {out_path}')

if __name__ == '__main__':
    main()
