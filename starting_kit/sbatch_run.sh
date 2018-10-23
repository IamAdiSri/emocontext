#!/bin/bash
#SBATCH -A aditya.srivastava
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --mincpus=12
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aditya.srivastava@research.iiit.ac.in

module load cuda/8.0 
module load cudnn/7-cuda-8.0 
python3 dbpedia_train.py cnn29 ./test_dbpedia/
