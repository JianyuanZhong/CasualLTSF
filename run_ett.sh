#!/bin/bash

seq_len=336
dataset=ETTh1

rm -rf log-$dataset

python main.py --dataset $dataset --seq_len $seq_len --pred_len 92 --epoch 200 --learning_rate 0.001
python main.py --dataset $dataset --seq_len $seq_len --pred_len 192 --epoch 200 --learning_rate 0.001
python main.py --dataset $dataset --seq_len $seq_len --pred_len 336 --epoch 200 --learning_rate 0.001
python main.py --dataset $dataset --seq_len $seq_len --pred_len 720 --epoch 200 --learning_rate 0.001

# dataset=ETTh2

# rm -rf log-$dataset

# python main.py --dataset $dataset --seq_len $seq_len --pred_len 92 
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 192 
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 336 
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 720 

# dataset=ETTm1

# rm -rf log-$dataset

# python main.py --dataset $dataset --seq_len $seq_len --pred_len 92  --epoch 100 --num_rnn_layer 5 --learning_rate 0.001
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 192 --epoch 100 --num_rnn_layer 5 --learning_rate 0.001
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 336  --epoch 100 --num_rnn_layer 5 --learning_rate 0.001
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 720  --epoch 100 --num_rnn_layer 5 --learning_rate 0.001

# dataset=ETTm2

# rm -rf log-$dataset

# python main.py --dataset $dataset --seq_len $seq_len --pred_len 92  --epoch 100 --num_rnn_layer 5 --learning_rate 0.001
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 192 --epoch 100 --num_rnn_layer 5 --learning_rate 0.001
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 336  --epoch 100 --num_rnn_layer 5 --learning_rate 0.001
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 720  --epoch 100 --num_rnn_layer 5 --learning_rate 0.001