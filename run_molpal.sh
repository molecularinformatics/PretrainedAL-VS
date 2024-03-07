#!/bin/bash
#SBATCH --job-name=pretrainedAL
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192GB
#SBATCH --time=14-00:00:00
#SBATCH --gpus-per-task=0
#SBATCH -e run/slurm_log/molpal_stderr.txt
#SBATCH -o run/slurm_log/molpal_stdout.txt

module load Anaconda3
conda activate yourENV
molpal run --config config.yaml
