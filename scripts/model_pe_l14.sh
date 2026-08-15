# Model-specific config — Meta PE-Core-L14-336 (671M, 336px)
MODEL="PE-Core-L14-336"
PRETRAINED="timm_openclip"
BATCH_SIZE=96
ACCUM_FREQ=1          # effective batch = 96 (2 GPUs × 48 × 1)
LR=1e-5
WARMUP=100
EPOCHS=5
XAC_LOSS_SCALE=0.01    # 3× paper default — 577 tokens vs 197
NPROC=2                # both GPUs (DDP — FSDP corrupts scalar params on 671M)
TORCH_COMPILE=1