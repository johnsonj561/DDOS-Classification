#!/bin/bash

#SBATCH -J ros_0pt0246_d2
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p shortq-bigdata7
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

module load anaconda/anaconda3/5.0.1

source activate jjohn273-tf.latest

srun python ../ros-rus-learner.py       \
          ros_rate=0.0246               \
          runs=10                       \
          width=32                      \
          depth=2                       \
          epochs=100                    \
          threshold_interval=0.0005
