#!/bin/bash
#SBATCH --job-name=valence_baseline
#SBATCH --account=ec403
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x-%j.out

mkdir -p logs
cd $SLURM_SUBMIT_DIR/..

python scripts/baseline_tfidf.py

