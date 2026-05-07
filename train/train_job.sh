#!/bin/bash
#SBATCH --job-name=mltrain1
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
cd "$WORKDIR"

export TORCH_HOME="$WORKDIR/.cache/torch"

echo "=============================================="
echo "Running on host: $(hostname)" 
echo "Working directory: $WORKDIR"
echo "Virtual environment: $VENV_PATH"
echo "Start time: $(date)"
echo "=============================================="

scontrol show job $SLURM_JOB_ID

module load cuda/11.8.0-gcc-13.2.0-oz34nbl

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "Missing virtual environment at $VENV_PATH"
    echo "Run with BOOTSTRAP_VENV=1 ./run_supercomputer.sh to create it automatically."
    exit 1
fi

source "$VENV_PATH/bin/activate"
torchrun --nproc_per_node=4 train.py

echo "End time: $(date)"