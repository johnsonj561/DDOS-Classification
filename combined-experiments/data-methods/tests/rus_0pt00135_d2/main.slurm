#!/bin/bash

#SBATCH -J rus_0pt00135_d2
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p shortq-bigdata7
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

module load anaconda/anaconda3/5.0.1

source activate jjohn273-tf.latest

srun python ../ros-rus-learner-test.py  \
          rus_rate=0.00135               \
          runs=30                       \
          width=32                      \
          depth=2                       \
          epochs=100                    \
          decision_threshold=0.3609
