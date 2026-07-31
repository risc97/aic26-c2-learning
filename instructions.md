# How to Replicate the Final C2LIP Model

## Overview

C2LIP (Concept Centric CLIP) fine-tunes a pre-trained **SigLIP ViT-B/16** model on the **CC3M** dataset with new captions from **DreamLIP**, using two novel losses:

- **NPC Loss** (Noun-Phrase Contrastive): contrasts images against individual noun-phrase concepts from their captions
- **XAC Loss** (Cross-Attended Concept): uses cross-attention between image patch tokens and noun-phrase embeddings

The architecture and forward pass are identical to a standard SigLIP model, so you can use standard evaluation pipelines.

---

## Step 1: Install Dependencies

```bash
pip install -r requirements-training.txt
```

Key dependencies: `torch>=1.9.0`, `torchvision`, `webdataset>=0.2.5`, `transformers<=4.57.6`, `timm>=0.9.8`, `huggingface_hub`, plus the base requirements in `requirements.txt`.

---

## Step 2: Download Data

You need both CC3M images and DreamLIP captions. Choose **one** of two methods for images:

### Option A — Direct URL download (many URLs may be dead)

1. Download `Train_GCC-training.tsv` from the [Conceptual Captions website](https://ai.google.com/research/ConceptualCaptions/download)
2. Run:
```bash
python data_processing/download_images_direct.py \
    --hf-cache-dir <HF_CACHE_DIR> \
    --image-dir <WHERE_TO_SAVE_IMAGES> \
    --cc3m-tsv <PATH_TO_TSV_FILE> \
    --output-pkl <PATH_TO_SAVE_METADATA_PICKLE>
```

### Option B — From HuggingFace (recommended, more complete)

```bash
python data_processing/download_images_cc3m_wds.py \
    --hf-cache-dir /workspace/data/hf_cache \
    --image-dir /workspace/data/images \
    --output-pkl /workspace/data/metadata/cc3m_metadata.pkl
```
This sources from [pixparse/cc3m-wds](https://huggingface.co/datasets/pixparse/cc3m-wds) on HuggingFace. Processes 576 tar shards individually with per-shard error handling and checkpointing — safe to resume if interrupted.

The captions come from [qidouxiong619/dreamlip_long_captions](https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions) — the scripts use the **`shortLLA_captions`** key (not the long captions from the DreamLIP paper).

**Expected output:** ~2.8M image-caption pairs, ~65 GB of images, ~4 GB metadata PKL, ~56 GB HF cache.

**Post-download cleanup:** After the PKL is built, the HF DreamLIP cache (56 GB) is no longer needed — the relevant captions are already in the PKL. Delete it:
```bash
rm -rf /workspace/data/hf_cache
```

**Final data footprint:** ~65 GB images + ~6.4 GB PKL (after Step 3) = ~71 GB total.

---

## Step 3: Extract Noun-Phrase Concepts

```bash
python data_processing/extract_nounphrases.py \
    --metadata-pkl-file /workspace/data/metadata/cc3m_metadata.pkl \
    --gpu 0 \
    --overwrite-metadata-file \
    --batch-size 192 \
    --max-new-tokens 50
```

This uses **Qwen3.5-0.8B** (a 0.8B-param instruct model) to extract noun phrases from captions and adds a `"nounphrases"` attribute to each sample's metadata in the pickle file. The script includes several performance optimizations for a single-GPU setup:

| Optimization | Effect |
|---|---|
| `torch.compile(mode="reduce-overhead")` | CUDA graph capture for the autoregressive decode loop (~30% faster) |
| Flash Attention 2 (`flash_attn`) | Lower VRAM, faster attention (auto-falls back to SDPA if not installed) |
| FP16 inference + TF32 matmul | Native tensor-core throughput on Ampere+ GPUs |
| Checkpointing every 100 batches | Safe to interrupt and resume |
| RAM-efficient design | PKL loaded once, captions extracted to lightweight list, then full dict freed — only ~3 GB RAM usage during inference |

**Model choice:** The paper used a proprietary Apple-internal NLP pipeline. The public release originally shipped with Phi-3.5-mini-instruct, but Qwen3.5-0.8B was chosen for this replication because it is ~5× smaller (0.8B vs 3.8B params) while producing equally good noun-phrase extractions — averaging 3.2 noun phrases per caption across 2.9M samples, with only 100 empty results (0.003%).

**Expected runtime:** ~40 hours on an RTX 5060 Ti (16 GB) at batch size 192, or ~100 hours on an RTX 3060 (12 GB). With the `_np_checkpoint.pkl` resume file, the run can be safely interrupted and continued.

**Result:** The PKL grows from ~4 GB to ~6.4 GB. Example outputs:

| Caption | Nounphrases |
|---------|-------------|
| *A white house with a red chimney is surrounded by a wooden fence.* | `white house`, `red chimney`, `wooden fence` |
| *A man with long hair and a black shirt playing a guitar.* | `man`, `guitar` |
| *A woman in a pink dress stands in front of a green backdrop.* | `woman`, `pink dress`, `green backdrop` |

---

## Step 4 (Optional): Pack Data into WebDataset Format

Use this if you want to transfer data between machines or prefer the webdataset training path:

```bash
python data_processing/pack_data.py \
    --image-root-dir <WHERE_TO_SAVE_IMAGES> \
    --metadata-pkl-file <PATH_TO_METADATA_PICKLE> \
    --output-wds-dir <WDS_ROOT_DIR>
```

This creates `.tar` shards for webdataset-style training (2000 samples per shard by default).

---

## Step 5: Launch Training

You have two options depending on your data format.

### If you packed data into WebDataset (Step 4)

Edit [scripts/train_full_model_wds_data.sh](scripts/train_full_model_wds_data.sh):
1. Set `train_data='/path/to/cc3m_np_wds/{00000..01411}.tar'` to your WDS shards path
2. Set `train_num_samples=2823019` (CC3M has ~2.82M samples)

Then:
```bash
cd scripts
bash train_full_model_wds_data.sh
```

### If using the pickle + image directory directly

Edit [scripts/train_full_model_pkl_data.sh](scripts/train_full_model_pkl_data.sh):
1. Set `train_data='/path/to/pkl/file/cc3m_np.pkl'`
2. Set `train_image_dir='/path/to/image/dir'`

Then:
```bash
cd scripts
bash train_full_model_pkl_data.sh
```

---

## Training Hyperparameters

The scripts use these key settings:

| Parameter | Value |
|-----------|-------|
| Model | `ViT-B-16-SigLIP` |
| Pretrained weights | `webli` (WebLI pre-training) |
| Learning rate | `1e-5` |
| Weight decay | `0.1` |
| Batch size | `96` per GPU |
| Epochs | `5` |
| Optimizer | AdamW (β₁=0.9, β₂=0.98, ε=1e-6) |
| Precision | `bf16` |
| Warmup steps | `50` |
| NPC loss scale | `1.0` |
| XAC loss scale | `0.01` |
| Distributed | 8 GPUs (`--nproc_per_node 8`) |

The effective global batch size is 96 × 8 = **768**.

---

## Key Flags That Enable C2LIP

All three must be enabled together for the full C2LIP model:

- `--siglip` — use SigLIP (sigmoid) loss instead of softmax contrastive
- `--output-tokens` — output image patch tokens from the vision encoder (required for XAC)
- `--npc-loss` — enable Noun-Phrase Contrastive loss
- `--xac-loss` — enable Cross-Attended Concept loss

When both `--npc-loss` and `--xac-loss` are on, the code automatically uses `C2LIP_Loss` (in [src/open_clip/loss.py](src/open_clip/loss.py)), which combines:

1. **Standard SigLIP contrastive loss** between full image and text embeddings
2. **NPC loss**: sigmoid loss on image vs. individual noun-phrase embeddings
3. **XAC loss**: cross-attention from noun-phrases over image tokens, then sigmoid loss

---

## Step 6: Evaluation

Since the trained model has the same architecture and forward pass as a standard SigLIP model, you can reuse existing evaluation pipelines (e.g., OpenCLIP's zero-shot eval, retrieval benchmarks, compositionality benchmarks).

The training code already supports:
- Zero-shot ImageNet evaluation (`--imagenet-val`, `--imagenet-v2`)
- Validation loss on a held-out set (`--val-data`)
- Logging to TensorBoard and/or Weights & Biases

---

## Hardware Requirements

The training script assumes **8 GPUs** with `bf16` precision. The model is ViT-B/16, so each GPU needs enough memory for a batch of 96 images + associated text tokens + gradient storage. Modern GPUs with 24–80 GB VRAM (A10, A100, H100, etc.) should be sufficient.

---

## Source Code Architecture

The codebase is a fork of **Open_CLIP** with C2LIP-specific additions:

### Core CLIP modules (`src/open_clip/`)
- `factory.py` — Model creation, loss creation (`create_loss`), tokenizer creation
- `model.py` — `CLIP` and `CustomTextCLIP` classes with `output_tokens` support
- `loss.py` — All loss functions including `C2LIP_Loss` (the core contribution)
- `transformer.py` — `VisionTransformer` (with `output_tokens`), `TextTransformer`
- `timm_model.py` — Timm model wrapper for SigLIP vision backbone
- `transform.py` — Image preprocessing pipeline
- `tokenizer.py` — Tokenizer implementations
- `pretrained.py` — Pretrained model download

### Training framework (`src/training/`)
- `main.py` (at `src/main.py`) — Entry point, orchestrates distributed training loop
- `train.py` — `train_one_epoch` and `evaluate` functions
- `data.py` — All dataloaders: `CC3m_NP_PKL_Dataset`, `get_cc3m_custom_np_wds_dataset`, collate functions
- `params.py` — Argument parser with all CLI flags
- `distributed.py` — Distributed training utilities
- `scheduler.py` — LR schedulers (cosine, const, const-cooldown)
- `zero_shot.py` — Zero-shot ImageNet evaluation
- `precision.py` — Automatic mixed precision (AMP) handling

---

## Expected Outputs

- **Checkpoints:** Saved to `./logs/{experiment_name}/checkpoints/epoch_{N}.pt` at each epoch, containing model state_dict, optimizer state, epoch number, and scaler state.
- **TensorBoard logs:** Saved to `./logs/{experiment_name}/tensorboard/`
- **Log files:** Saved to `./logs/{experiment_name}/out.log`
- **Experiment name format:** `C2LIP-lr-{lr}-wd-{wd}-bs-{bs}-ep-{epochs}-npc-loss-scale-{npc_loss_scale}-xac-loss-scale-{xac_loss_scale}`

---

## Checkpoints

The paper's checkpoints are not publicly downloadable — you need to reach out to the authors directly.

---

## Citation

If you find this work useful, please cite:

```
@inproceedings{c2lip_cvpr26,
  title   = {No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models},
  author  = {Hai Xuan Pham and David Hoffmann and Ricardo Guerrero and Brais Martinez},
  booktitle = {CVPR},
  year    = {2026}
}
```
