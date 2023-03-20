#!/bin/bash
#SBATCH --job-name=Grid_336
#SBATCH --mail-user=jianyuanzhong@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1

# change cuda version to 11
export PATH=$PATH:/usr/local/cuda-11/bin
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64

echo "Driver Cuda Version"
nvidia-smi

echo "Local Cuda Version"
nvcc -V

# activate conda env
echo "activate conda env"
conda init bash
source activate traffic

seq_len=336
# dataset=ETTh1

# rm -rf log-$dataset

# python tune.py --dataset $dataset --seq_len $seq_len --pred_len 96 
# python tune.py --dataset $dataset --seq_len $seq_len --pred_len 192 
# python tune.py --dataset $dataset --seq_len $seq_len --pred_len 336 
# python tune.py --dataset $dataset --seq_len $seq_len --pred_len 720 

dataset=ETTh2

rm -rf log-$dataset-seq$seq_len

python tune.py --dataset $dataset --seq_len $seq_len --pred_len 96 
python tune.py --dataset $dataset --seq_len $seq_len --pred_len 192 
python tune.py --dataset $dataset --seq_len $seq_len --pred_len 336 
python tune.py --dataset $dataset --seq_len $seq_len --pred_len 720 

# dataset=ETTm1

# rm -rf log-$dataset-seq$seq_len

# python tune.py --dataset $dataset --seq_len $seq_len --pred_len 96 
# python tune.py --dataset $dataset --seq_len $seq_len --pred_len 192
# python tune.py --dataset $dataset --seq_len $seq_len --pred_len 336 
# python tune.py --dataset $dataset --seq_len $seq_len --pred_len 720 

# dataset=ETTm2

# # seq_len=512
# rm -rf log-$dataset-seq$seq_len

# python tune.py --dataset $dataset --seq_len $seq_len --pred_len 96 
# python tune.py --dataset $dataset --seq_len $seq_len --pred_len 192
# python tune.py --dataset $dataset --seq_len $seq_len --pred_len 336 
# python tune.py --dataset $dataset --seq_len $seq_len --pred_len 720 