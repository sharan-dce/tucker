#!/bin/bash

#SBATCH --time=24:00:00

#SBATCH --job-name=sneaky

#SBATCH --gres=gpu:p100:1

#SBATCH --account=engs-tvg

# module load anaconda3/2019.03
# module load gpu/cuda/9.0.176
module load gpu/cuda/10.1.243
# module load gpu/cuda/10.2.89

# module load gpu/cudnn/7.3.1__cuda-9.0
module load gpu/cudnn/7.6.5__cuda-10.1
# module load gpu/cuda/10.0.130 
# module load gpu/cudnn/7.5.0__cuda-10.0
# module load gpu/cuda/9.2.148
# module load gpu/cudnn/7.6.3__cuda-9.2
# module load anaconda3/2019.03 

# module load gpu/cuda/9.0.176

# module load gpu/cudnn/7.3.1__cuda-9.0

source activate sneaky_env


# source activate sneaky_env

python3 -u train_script.py\
    --model=tucker\
    --datapath=../data/FB15k\
    --num_iterations=500\
    --batch_size=64\
    --lr=0.003\
    --dr=0.99\
    --edim=200\
    --rdim=200\
    --cuda=True\
    --input_dropout=0.2\
    --hidden_dropout1=0.2\
    --hidden_dropout2=0.3\
    --label_smoothing=0.0 > FB15k_training_result.txt


