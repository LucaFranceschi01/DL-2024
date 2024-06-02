#!/bin/bash
#SBATCH -J DL-lf-first_try
#SBATCH -p short
#SBATCH -N 1
#SBATCH --chdir=/home/lfranceschi/final
#SBATCH --gres=gpu:1
#SBATCH --time=10:00

#SBATCH -o slurm.%N.%J.%u.out # STDOUT
#SBATCH -e slurm.%N.%J.%u.err # STDERR

module load CUDA/11.4.3

module load Python/3.8.6-GCCcore-10.2.0

# python -m venv .venv

source .venv/bin/activate

# pip install -r cluster-requirements.txt

# jupyter nbconvert --to script FinalProject.ipynb

python FinalProject.py

deactivate
