#!/bin/bash
#SBATCH -t 1-00
#SBATCH -N 1  # nodes
#SBATCH -n 1  # tasks
#SBATCH -c 16  # cores per task
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=64G
#SBATCH --mail-user=yqin3@nd.edu
#SBATCH --mail-type=END
#SBATCH -o ./log/tiny/%x_%a.log
#SBATCH -J r18_7.9  # job name
#SBATCH --array=1

source /home/ywang144/miniconda3/etc/profile.d/conda.sh
conda activate nf
python res18_main.py --mode tnt --type correct --dataset tiny --var1 0.4 --var2 0.4 --device RRAM1 \
--num $SLURM_ARRAY_TASK_ID --mark 7.9



