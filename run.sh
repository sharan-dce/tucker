#!/bin/bash

#SBATCH --time=24:00:00

#SBATCH --job-name=sneaky

#SBATCH --partition=htc

#SBATCH --gres=gpu:1

#SBATCH --account=engs-tvg
 

module load anaconda3/2019.03

module load gpu/cuda/9.0.176

module load gpu/cudnn/7.3.1__cuda-9.0


source activate sneakyenv


python3 sneaky_overfit  --overfit_model \
                        --architecture resnet20 \
                        --dataset cifar10 \
                        --epochs 300 \
                        --classes 10 \
                        --flip_budgets 0 \
                        --batch_size 128 \
                        --save_path $SHARDATA/saved_models/cifar-clean \
                        --strategy random \
                        --random_seed 0 \
                        --norm 2 \
                        --gammas 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 \
                        # norm and gamma parameters to be ignored \

