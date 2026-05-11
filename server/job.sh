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

cd "$WORKDIR"

# ssh -L 8765:${COMPUTE_NODE}:8765 $USER@131.91.163.212
# WebSocket endpoint: ws://localhost:8765

echo " Slurm Job ID  : $SLURM_JOB_ID"
echo " Compute node  : $COMPUTE_NODE"
echo " Working dir   : $WORKDIR"
echo " Venv          : $VENV_PATH"
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