#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=sneaky
#SBATCH --gres=gpu:p100:1
#SBATCH --account=engs-tvg
 
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.6.5__cuda-10.1

source activate sneaky_env

python3 -u train_script.py\
    --model=tucker\
    --datapath=../data/FB15k-237\
    --num_iterations=600\
    --batch_size=128\
    --lr=0.0005\
    --dr=1.0\
    --edim=200\
    --rdim=200\
    --cuda=True\
    --input_dropout=0.3\
    --hidden_dropout1=0.4\
    --hidden_dropout2=0.5\
    --label_smoothing=0.1 > FB15k-237_training_result.txt
