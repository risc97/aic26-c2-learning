# Model-specific config — Meta PE-Core-G14-448 (2.4B, 448px)
MODEL="PE-Core-G14-448"
PRETRAINED="timm_openclip"
BATCH_SIZE=1           # 2.4B — OOMs above 1 at 448px
ACCUM_FREQ=8           # effective batch = 16 (2 GPUs × 1 × 8)
# G14 at native 448px needs 40GB+ GPUs — train L14 instead on 2×20GB
LR=1e-5                # same as L14 — AdamW normalizes per-parameter step size
WARMUP=200
EPOCHS=5
XAC_LOSS_SCALE=0.06    # 6× paper default — 1025 tokens vs 197
NPROC=2                # use both GPUs
FSDP=1                 # FSDP — DDP can't fit 2.4B on 2× 20GB
FSDP_CPU_OFFLOAD=1      # offload params/optimizer to CPU RAM — frees ~9GB/GPU
TORCH_COMPILE=0