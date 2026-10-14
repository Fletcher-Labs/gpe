#!/bin/bash
#SBATCH --output=results/%x/output_%x.txt        # Standard output and error log (%j will be replaced with the job ID)
#SBATCH --error=results/%x/error_%x.txt          # Error log
#SBATCH --ntasks=1                    # Run on a single task
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem=16G                      # Memory per node
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1


# Activate your Conda environment
source ~/.bashrc
conda activate /home/rubaiyae/.conda/envs/gpenv

# Run the Python script
python /orcd/home/002/rubaiyae/gpe/LHY_adss_Nscan.py $SLURM_JOB_NAME