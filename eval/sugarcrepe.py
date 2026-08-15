#!/usr/bin/env python3
"""Original SugarCrepe evaluation — binary ITT: sim(img, pos) > sim(img, neg)?"""
import json, os, sys, time, torch, numpy as np
sys.path.insert(0, 'src')
from PIL import Image
from open_clip.factory import create_model, get_model_config, get_tokenizer
from open_clip.transform import image_transform_v2, PreprocessCfg

SUBSETS = ['add_att', 'add_obj', 'replace_att', 'replace_obj', 'replace_rel', 'swap_att', 'swap_obj']
COCO_DIR = 'data/coco/val2017'
SC_DATA = '/tmp/sugarcrepe_repo/data'
BATCH = 256

def load_model(ckpt_path, model_name, device):
    cfg = get_model_config(model_name)
    kw = {}
    if 'init_logit_bias' in cfg:
        kw['init_logit_scale'] = np.log(10)
        kw['init_logit_bias'] = cfg['init_logit_bias']
    kw['output_tokens'] = True
    model = create_model(model_name, pretrained='', precision='fp32', device=device, output_dict=True, **kw)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    if any(k.startswith('module.') for k in sd): sd = {k[7:]: v for k, v in sd.items()}
    if any(k.startswith('_orig_mod.') for k in sd): sd = {k[10:]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    tokenizer = get_tokenizer(model_name)
    return model, tokenizer

def build_pp(model, siglip_pp):
    if siglip_pp:
        return image_transform_v2(PreprocessCfg(
            mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), size=224, interpolation='bicubic', resize_mode='squash'), is_train=False)
    else:
        pp = model.visual.preprocess_cfg
        if isinstance(pp, dict):
            s = pp.get('size', (336,336)); s = s[0] if isinstance(s,(tuple,list)) else s
            pp = PreprocessCfg(mean=pp.get('mean'), std=pp.get('std'), size=s,
                               interpolation=pp.get('interpolation','bicubic'),
                               resize_mode=pp.get('resize_mode','shortest'))
        return image_transform_v2(pp, is_train=False)

def eval_sugarcrepe(model, tokenizer, preprocess, device, label):
    # Load all data
    all_samples = []
    for sub in SUBSETS:
        with open(f'{SC_DATA}/{sub}.json') as f:
            for sid, s in json.load(f).items():
                s['subset'] = sub
                all_samples.append(s)
    print(f'\n  [{label}] {len(all_samples)} samples')

    # Unique images
    unique_imgs = list(set(s['filename'] for s in all_samples))
    img_feats = {}
    for i in range(0, len(unique_imgs), BATCH):
        batch = []
        for fn in unique_imgs[i:i+BATCH]:
            img = Image.open(f'{COCO_DIR}/{fn}').convert('RGB')
            batch.append(preprocess(img))
        images = torch.stack(batch).to(device)
        with torch.no_grad():
            r = model.encode_image(images, normalize=True)
            feats = r[0] if isinstance(r, tuple) else r
        for fn, feat in zip(unique_imgs[i:i+BATCH], feats.cpu()):
            img_feats[fn] = feat

    # Unique captions
    cap_idx = {}
    all_caps = []
    for s in all_samples:
        for key in ['caption', 'negative_caption']:
            ck = (s['filename'], s['subset'], key)
            if ck not in cap_idx:
                cap_idx[ck] = len(all_caps)
                all_caps.append(s[key])
    text_feats = [None] * len(all_caps)
    for i in range(0, len(all_caps), BATCH):
        tok = tokenizer(all_caps[i:i+BATCH])
        if isinstance(tok, dict): tok = {k: v.to(device) for k, v in tok.items()}
        else: tok = tok.to(device)
        with torch.no_grad():
            feats = model.encode_text(tok, normalize=True)
            for j, feat in enumerate(feats.cpu()):
                text_feats[i+j] = feat

    # Score
    results = {}
    for sub in SUBSETS:
        ss = [s for s in all_samples if s['subset'] == sub]
        correct = 0
        for s in ss:
            img_f = img_feats[s['filename']]
            pos_f = text_feats[cap_idx[(s['filename'], sub, 'caption')]]
            neg_f = text_feats[cap_idx[(s['filename'], sub, 'negative_caption')]]
            if (img_f @ pos_f).item() > (img_f @ neg_f).item():
                correct += 1
        results[sub] = {'correct': correct, 'total': len(ss), 'acc': 100.0*correct/len(ss)}

    total_c = sum(r['correct'] for r in results.values())
    total_t = sum(r['total'] for r in results.values())
    results['OVERALL'] = {'correct': total_c, 'total': total_t, 'acc': 100.0*total_c/total_t}
    return results

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES','') != '' else 'cpu')
    print(f'Device: {device}')

    configs = [
        ('PE-Core-L14-336', 'checkpoints/PE-Core-L14-336-0508-0854/checkpoints/epoch_2.pt', False),
        ('ViT-B-16-SigLIP', 'checkpoints/reference/c2lip_npc_xac.pt', True),
    ]

    all_results = {}
    for model_name, ckpt_path, siglip_pp in configs:
        t0 = time.time()
        model, tokenizer = load_model(ckpt_path, model_name, device)
        pp = build_pp(model, siglip_pp)
        results = eval_sugarcrepe(model, tokenizer, pp, device, model_name)
        elapsed = time.time() - t0
        all_results[model_name] = results

        # Print
        sep = '=' * 55
        print(f'\n{sep}')
        print(f'{model_name} - SugarCrepe ITT ({elapsed:.0f}s)')
        print(sep)
        for sub in SUBSETS + ['OVERALL']:
            r = results[sub]
            m = ' <<<' if sub == 'OVERALL' else ''
            print(f"  {sub:<20} {r['correct']:>6}/{r['total']:<6} {r['acc']:.1f}%{m}")

    # Save
    out_path = 'checkpoints/sugarcrepe_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved to {out_path}')
