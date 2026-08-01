#!/bin/bash
# Remove all experiment directories from checkpoints/
# Usage: bash scripts/clean-checkpoints.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Cleaning checkpoints/ ..."
rm -rf checkpoints/*/
echo "Done."
ls checkpoints/ 2>/dev/null || echo "(empty)"
