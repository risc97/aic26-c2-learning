#!/bin/bash
# =============================================================================
# C2LIP Training — single-GPU with all performance optimizations baked in
# =============================================================================
# Usage:
#   bash scripts/train_c2lip.sh scripts/model_siglip_b16.sh
#   bash scripts/train_c2lip.sh scripts/model_pe_l14.sh
# =============================================================================
set -euo pipefail

MODEL_CONFIG="${1:?Usage: $0 <model_config> (e.g. scripts/model_siglip_b16.sh)}"
source "$MODEL_CONFIG"

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Shared training hyperparameters (model-agnostic)
# ---------------------------------------------------------------------------
PRECISION="bf16"          # Blackwell native; also fine on Ampere
WEIGHT_DECAY=0.1
BETA1=0.9
BETA2=0.98
EPS=1e-6
SEED=42

# ---------------------------------------------------------------------------
# C2LIP loss scales
# ---------------------------------------------------------------------------
NPC_LOSS_SCALE=1.0
XAC_LOSS_SCALE=0.01

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
TRAIN_DATA="/workspace/data/metadata/cc3m_metadata.pkl"
IMAGE_DIR="/workspace/data/images"
DATASET_TYPE="cc3m_custom_np_pkl"

# ---------------------------------------------------------------------------
# Performance optimizations (baked in)
# ---------------------------------------------------------------------------
NUM_WORKERS=4             # async CPU preprocessing — GPU never starved
TORCH_COMPILE=1            # works with 256px + grad checkpointing on 16GB

# ---------------------------------------------------------------------------
# Run name (timestamped — re-runs never collide)
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%d%m-%H%M)
EFFECTIVE_BS=$((BATCH_SIZE * ACCUM_FREQ))
RUN_NAME="${MODEL}-${TIMESTAMP}"

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
echo "============================================================================"
echo "C2LIP Training"
echo "  Model:      $MODEL  |  Pretrained: $PRETRAINED"
echo "  Precision:  $PRECISION  |  Compile: $([ $TORCH_COMPILE -eq 1 ] && echo yes || echo no)"
echo "  Batch:      $BATCH_SIZE × $ACCUM_FREQ = $EFFECTIVE_BS"
echo "  LR:         $LR  |  Warmup: $WARMUP  |  Epochs: $EPOCHS"
echo "  Run:        checkpoints/$RUN_NAME/"
echo "============================================================================"

# ---------------------------------------------------------------------------
# Save human-readable notes alongside the run
# ---------------------------------------------------------------------------
RUN_DIR="./checkpoints/$RUN_NAME"
mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/notes.txt" << EOF
C2LIP Training Run
==================
Date:     $(date)
Model:    $MODEL
Pretrained: $PRETRAINED
Precision:  $PRECISION
Torch Compile: $([ $TORCH_COMPILE -eq 1 ] && echo yes || echo no)

Hyperparameters:
  Batch size:   $BATCH_SIZE × $ACCUM_FREQ = $EFFECTIVE_BS
  LR:           $LR
  Weight decay: $WEIGHT_DECAY
  Warmup:       $WARMUP
  Epochs:       $EPOCHS
  Adam:         β₁=$BETA1 β₂=$BETA2 ε=$EPS
  Seed:         $SEED

C2LIP Losses:
  NPC loss scale: $NPC_LOSS_SCALE
  XAC loss scale: $XAC_LOSS_SCALE

Data:
  Images:    $IMAGE_DIR
  Metadata:  $TRAIN_DATA
  Dataset:   $DATASET_TYPE
EOF

echo "Notes saved to $RUN_DIR/notes.txt"

torchrun --nproc_per_node 1 \
    src/main.py \
    --model "$MODEL" \
    --pretrained "$PRETRAINED" \
    --precision "$PRECISION" \
    --batch-size "$BATCH_SIZE" \
    --accum-freq "$ACCUM_FREQ" \
    --lr "$LR" \
    --wd "$WEIGHT_DECAY" \
    --warmup "$WARMUP" \
    --epochs "$EPOCHS" \
    --beta1 "$BETA1" \
    --beta2 "$BETA2" \
    --eps "$EPS" \
    --seed "$SEED" \
    --workers "$NUM_WORKERS" \
    --train-data "$TRAIN_DATA" \
    --images-dir-path "$IMAGE_DIR" \
    --dataset-type "$DATASET_TYPE" \
    --dataset-resampled \
    --save-frequency 1 \
    --logs ./checkpoints \
    --report-to tensorboard \
    --siglip \
    --output-tokens \
    --grad-checkpointing \
    --npc-loss \
    --npc-loss-scale "$NPC_LOSS_SCALE" \
    --xac-loss \
    --xac-loss-scale "$XAC_LOSS_SCALE" \
    --name "$RUN_NAME" \
    --delete-previous-checkpoint \
    $([ $TORCH_COMPILE -eq 1 ] && echo "--torchcompile")
