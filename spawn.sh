#!/bin/sh

sbatch -n1 train_FB15k-237.sh  
sbatch -n1 train_FB15k.sh      
sbatch -n1 train_WN18RR.sh     
sbatch -n1 train_WN18.sh
