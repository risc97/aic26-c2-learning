# Model-specific config — Meta PE-Core-G14-448 (2.4B, 448px)
MODEL="PE-Core-G14-448"
PRETRAINED="timm_openclip"
BATCH_SIZE=4           # 2.4B at 448px
ACCUM_FREQ=4           # effective batch = 32 (2 GPUs × 4 × 4)
LR=5e-6                # same as L14 — AdamW normalizes per-parameter step size
WARMUP=200
EPOCHS=5
XAC_LOSS_SCALE=0.05    # 5× paper default — 1025 tokens vs 197
NPROC=2                # use both GPUs
FSDP=1                 # FSDP — DDP can't fit 2.4B on 2× 24GB
TORCH_COMPILE=0