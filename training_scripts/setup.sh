#!/bin/bash

#SBATCH --time=24:00:00

#SBATCH --job-name=sneaky

#SBATCH --gres=gpu:1

#SBATCH --account=engs-tvg
module load anaconda3/2019.03 
module load gpu/cuda/9.0.176

module load gpu/cudnn/7.3.1__cuda-9.0

source activate sneakyenv

nvidia-smi

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
