#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=10:00:00
#SBATCH --mincpus=12
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aditya.srivastava@research.iiit.ac.in

module load cuda/8.0
module load cudnn/7-cuda-8.0 
module load openmpi/2.1.1-cuda8
python3 baseline.py -config testBaseline.config
