# Model-specific config — Meta PE-Core-G14-448 (2.2B, 448px)
MODEL="PE-Core-G14-448"
PRETRAINED="timm_openclip"
BATCH_SIZE=4           # 2.2B at 448px — FSDP required for >1 GPU
ACCUM_FREQ=8           # effective batch = 32
LR=5e-6                # same as L14 — AdamW normalizes per-parameter step size
WARMUP=200
EPOCHS=5
XAC_LOSS_SCALE=0.05    # 5× paper default — 1025 tokens vs 197
