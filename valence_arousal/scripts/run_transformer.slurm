#!/bin/bash
#SBATCH --job-name=transformer_emotion
#SBATCH --account=ec403
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p logs

cd /fp/homes01/u01/ec-andremg/work/exam/valence_arousal

module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
export TRANSFORMERS_OFFLINE=1

python scripts/train_transformer.py
