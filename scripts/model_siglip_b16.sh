# Model-specific config — SigLIP ViT-B-16 (the paper's model)
MODEL="ViT-B-16-SigLIP"
PRETRAINED="webli"
BATCH_SIZE=24
ACCUM_FREQ=4          # effective batch = 96 (matches paper)
LR=1e-5
WARMUP=50
EPOCHS=5
