#!/bin/bash
#SBATCH --job-name=BH
#SBATCH --mail-user=jianyuanzhong@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1

# change cuda version to 11
# export PATH=$PATH:/usr/local/cuda-11/bin
# export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64

echo "Driver Cuda Version"
nvidia-smi

echo "Local Cuda Version"
nvcc -V

# activate conda env
echo "activate conda env"
conda init bash
source activate traffic

# seq_len=336
# dataset=ETTh1

# rm -rf log-best-$dataset

# python main.py --dataset $dataset --seq_len $seq_len --pred_len 96 --config_pth best_configs/etth1_96.json
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 192 --config_pth best_configs/etth1_192.json
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 336 --config_pth best_configs/etth1_336.json
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 720 --config_pth best_configs/etth1_720.json

# dataset=ETTh2

# rm -rf log-best-$dataset

# python main.py --dataset $dataset --seq_len $seq_len --pred_len 96 --config_pth best_configs/etth2_96.json
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 192 --config_pth best_configs/etth2_192.json
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 336 --config_pth best_configs/etth2_336.json
# python main.py --dataset $dataset --seq_len $seq_len --pred_len 720 --config_pth best_configs/etth2_720.json


dataset=ETTm2
seq_len=512
rm -rf log-best-$dataset

python main.py --dataset $dataset --seq_len $seq_len --pred_len 96 --config_pth best_configs/ettm2_96.json
python main.py --dataset $dataset --seq_len 720 --pred_len 192 --config_pth best_configs/ettm2_192.json
python main.py --dataset $dataset --seq_len $seq_len --pred_len 336 --config_pth best_configs/ettm2_336.json
python main.py --dataset $dataset --seq_len $seq_len --pred_len 720 --config_pth best_configs/ettm2_720.json