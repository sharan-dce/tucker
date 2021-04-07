#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=sneaky
#SBATCH --gres=gpu:p100:1
#SBATCH --account=engs-tvg
 
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.6.5__cuda-10.1

source activate sneaky_env

python3 -u main.py\
    --model=tucker\
    --dataset=WN18\
    --num_iterations=500\
    --batch_size=128\
    --lr=0.005\
    --dr=0.995\
    --edim=200\
    --rdim=30\
    --input_dropout=0.2\
    --hidden_dropout1=0.1\
    --hidden_dropout2=0.2\
    --label_smoothing=0.1 > WN18_training_result.txt
