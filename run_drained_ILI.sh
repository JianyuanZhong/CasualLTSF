#!/bin/bash

seq_len=36
dataset=ILI

python train_Dlinear.py --dataset $dataset --seq_len $seq_len --pred_len 24 --batch_size 16 #--individual
python train_Dlinear.py --dataset $dataset --seq_len $seq_len --pred_len 36 --batch_size 16 #--individual
python train_Dlinear.py --dataset $dataset --seq_len $seq_len --pred_len 48 --batch_size 16 #--individual
python train_Dlinear.py --dataset $dataset --seq_len $seq_len --pred_len 60 --batch_size 16 #--individual