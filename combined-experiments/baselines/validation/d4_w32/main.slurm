#!/bin/bash

#SBATCH -J partd_d4_w32
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

module load anaconda/anaconda3/5.0.1

source activate jjohn273-tf.latest

srun python ../baseline-learner.py       \
          runs=5              \
          width=32            \
          depth=4             \
          epochs=100          \
          threshold_interval=0.0001     
