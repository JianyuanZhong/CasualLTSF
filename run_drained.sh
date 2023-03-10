#!/bin/bash

seq_len=336
dataset=ETTh2

rm -rf log-$dataset

python train_Dlinear.py --dataset $dataset --seq_len $seq_len --pred_len 92 #--individual
python train_Dlinear.py --dataset $dataset --seq_len $seq_len --pred_len 192 #--individual
python train_Dlinear.py --dataset $dataset --seq_len $seq_len --pred_len 336 #--individual
python train_Dlinear.py --dataset $dataset --seq_len $seq_len --pred_len 720 #--individual