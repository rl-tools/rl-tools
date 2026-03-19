#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=300G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --array=0-3

set -euo pipefail

variants=(base crossconv 2x 128)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"

echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Variant:   $variant"
echo "Node:      $(hostname)"
echo "CWD:       $(pwd)"

ulimit -n 65536

./build/src/rendering/raytracing/yaw_prediction/rendering_raytracing_yaw_prediction_${variant} \
  --scene-dir /scr/jonas/ai2thor-hab/glb \
  --num-load-threads 16 \
  --num-scenes 1000 \
  --num-val-scenes 100 \
  --num-iterations 2000000
