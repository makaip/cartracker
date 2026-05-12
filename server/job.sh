#!/bin/bash
#SBATCH --job-name=mltrain
#SBATCH --partition=shortq7-gpu   # Partition name
#SBATCH --gres=gpu:4              # Number of GPUs
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=28        # Number of CPU cores
#SBATCH --mem=64G                 # Memory allocation
#SBATCH --output=slurm-%j.out     # Standard output log
#SBATCH --error=slurm-%j.err      # Error log file

set -Eeuo pipefail

WORKDIR="${RUN_WORKDIR:-${SLURM_SUBMIT_DIR:-/mnt/beegfs/home/jpindell2022/ouri_project/mltests/traffictrack}}"
VENV_PATH="${VENV_DIR:-$WORKDIR/.venv}"
COMPUTE_NODE="$(hostname --fqdn 2>/dev/null || hostname)"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$WORKDIR/.cache/ultralytics}"
export TORCH_HOME="${TORCH_HOME:-$WORKDIR/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKDIR/.cache}"

cd "$WORKDIR"

# ssh -L 8765:${COMPUTE_NODE}:8765 $USER@131.91.163.212
# WebSocket endpoint: ws://localhost:8765

echo " Slurm Job ID  : $SLURM_JOB_ID"
echo " Compute node  : $COMPUTE_NODE"
echo " Working dir   : $WORKDIR"
echo " Venv          : $VENV_PATH"
echo " YOLO config   : $YOLO_CONFIG_DIR"
echo " TORCH home    : $TORCH_HOME"
echo " XDG cache     : $XDG_CACHE_HOME"
echo " Start time    : $(date --iso-8601=seconds)"

scontrol show job "$SLURM_JOB_ID" || true

module load cuda/11.8.0-gcc-13.2.0-oz34nbl

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "Missing virtual environment at $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"

python "$WORKDIR/server/main.py"

echo "End time: $(date)"