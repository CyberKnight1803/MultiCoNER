#!/bin/bash 

# SLURM SCRIPT

#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --job-name="modx"
#SBATCH -o outputs/slurm.%j.out
#SBATCH -e outputs/slurm.%j.err
#SBATCH --mail-user=f20190083@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=END
#SBATCH --export=ALL

# LOAD MODULES
module purge
module load gcc-9.3.0-gcc-8.3.1-4oza3xc
module load python-3.8.7-gcc-9.3.0-exf3jun
module load nccl-2.8.3-1-gcc-9.3.0-sazlqgr
module load cuda-11.1.1-gcc-9.3.0-cu2kblh
module load cudnn-8.0.4.30-11.1-gcc-9.3.0-erjm43z

# Activate Virtual Environment
source /home/aruna/omkar/envs/code_mixing/bin/activate 

# Run Program
cd /home/aruna/omkar/codemix/

srun ~/omkar/envs/code_mixing/bin/python main.py \
--workers 4 \
--epochs 8 \
--batch_size 32 \
--base_model "xlm-roberta-large" \
--tagging_scheme "bio" \
--seq_length 128 \
--name "code_mix" \
--dataset_name "twitter" \
--dataset "./datasets/Others"