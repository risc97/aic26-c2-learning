# Model-specific config — Meta PE-Core-L14-336 (671M, 336px)
MODEL="PE-Core-L14-336"
PRETRAINED="timm_openclip"
BATCH_SIZE=24
ACCUM_FREQ=4          # effective batch = 96
LR=5e-6               # lower than SigLIP — larger model, more sensitive
WARMUP=100
EPOCHS=5
