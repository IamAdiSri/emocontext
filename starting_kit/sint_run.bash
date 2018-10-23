module load cuda/8.0
module load cudnn/7-cuda-8.0 
module load openmpi/2.1.1-cuda8
python3 baseline.py -config testBaseline.config
