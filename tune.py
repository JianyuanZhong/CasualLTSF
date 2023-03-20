# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pytorch_lightning import seed_everything

import os
import logging
import time
import datetime
from tqdm import tqdm
import argparse

# Import model
from NLinear import Model
from model_drained import RNN
# Import functions
from utils import dataset_preparation, make_noise, metric

# set precision for rtx3090
torch.set_float32_matmul_precision('highest')


#print('Is GPU available? {}'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DomainGen_Graph")

datasets = ['ONP', 'Moons', 'MNIST', 'Elec2']
parser.add_argument("--dataset", default="Moons", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))

# Hyper-parameters
parser.add_argument("--noise_dim", default=768, type=int,
                    help="the dimension of the LSTM input noise.")
parser.add_argument("--num_rnn_layer", default=10, type=int,
                    help="the number of RNN hierarchical layers.")
parser.add_argument("--latent_dim", default=768, type=int,
                    help="the latent dimension of RNN variables.")
parser.add_argument("--hidden_dim", default=768, type=int,
                    help="the latent dimension of RNN variables.")
parser.add_argument("--noise_type", choices=["Gaussian", "Uniform"], default="Gaussian",
                    help="The noise type to feed into the generator.")
parser.add_argument("--beta", type=float, default=0.4)

parser.add_argument("--num_workers", default=0, type=int,
                    help="the number of threads for loading data.")
parser.add_argument("--epoches", default=100, type=int,
                    help="the number of epoches for each task.")
parser.add_argument("--batch_size", default=128, type=int,
                    help="the number of epoches for each task.")
parser.add_argument("--learning_rate", default=5e-3, type=float,
                    help="the unified learning rate for each single task.")

parser.add_argument("--is_test", default=True, type=bool,
                    help="if this is a testing period.")

# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels

args = parser.parse_args()
print(f"dim: {args.noise_dim}")

# setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
log_dir = f"log-{args.dataset}-seq{args.seq_len}"
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
log_file = '{}/log_pred{}_{}.log'.format(log_dir, args.pred_len, datetime.datetime.now().strftime("%Y_%B_%d_%I-%M-%S%p"))
open(log_file, 'a').close()

# create logger
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# add to log file
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

def log(str): logger.info(str)


log('Is GPU available? {}'.format(torch.cuda.is_available()))


def train(dataloader, optimizer, rnn_unit, args, task_id=0, input_E=None, input_hidden=None):
    E = input_E
    hidden = input_hidden
    log("Start Training on Domain {}...".format(task_id))
    objective = torch.nn.SmoothL1Loss(beta=args.beta)
    for epoch in range(args.epoches):
        mses = []
        with tqdm(dataloader, unit="batch") as tepoch:
            for X, Y, _, _ in tepoch:
                tepoch.set_description("Task_ID: {} Epoch {}".format(task_id, epoch))
                
                X, Y  = X.float().to(device), Y.float().to(device)
                Y = Y[:, -args.pred_len:, :]
                initial_noise = make_noise((1, args.noise_dim), args.noise_type).to(device)
                
                #  Training on Single Domain
                rnn_unit.train()
                optimizer.zero_grad()
                E, hidden, pred = rnn_unit(X, initial_noise, E, hidden)
                E = E.detach()
                hidden = tuple([i.detach() for i in hidden])
                loss = objective(pred, Y)
                
                #prediction = torch.as_tensor(pred.detach()).float()
                #mse = torch.nn.MSELoss(prediction.squeeze(-1), Y)
                # Backward and optimize
                # optimizer.zero_grad()
                loss.backward()
                optimizer.step()    
                mses.append(loss.item())
                tepoch.set_postfix(loss=(sum(mses) / len(mses)))
            
            
            #log("Task_ID: {}\tEpoch: {}\tAverage Training Accuracy: {}".format(task_id, epoch, np.mean(accs)))
    return E, hidden, rnn_unit
    

def evaluation(dataloader, rnn_unit, args, input_E, input_hidden):
    rnn_unit.eval()
    E = input_E
    hidden = input_hidden
    preds = []
    trues = []
    test_size = 0
    log("Start Testing...")
    with tqdm(dataloader, unit="batch") as tepoch:
        for X, Y, _, _ in tepoch:
            test_size += X.size(0)                
            X, Y  = X.float().to(device), Y.float().to(device)
            Y = Y[:, -args.pred_len:, :]
            initial_noise = make_noise((1, args.noise_dim), args.noise_type).to(device)
            with torch.no_grad():
                _, _, pred = rnn_unit(X, initial_noise, E, hidden)
                # loss = F.mse_loss(pred, Y, reduction='sum')
                # mae = F.l1_loss(pred.squeeze(), Y, reduction='sum')
                pred = pred.cpu().numpy()
                true = Y.cpu().numpy()
                preds.append(pred)
                trues.append(true)

                # tepoch.set_postfix(loss=mse.item(), mae=mae.item())

    preds = np.concatenate(preds, axis=0).reshape(-1, pred.shape[-2], pred.shape[-1])
    trues = np.concatenate(trues, axis=0).reshape(-1, pred.shape[-2], pred.shape[-1])

    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    log("Testing MSE is {}".format(mse))
    log("Testing MAE is {}".format(mae))
    
    return mse, mae


def trail(args):
    # reset the state of random generators 
    # for reproducibility
    seed_everything(123, workers=True)
    
    log('use {} data'.format(args.dataset))
    log('-'*40)
    
    if args.dataset.startswith('ETTh'):
        num_tasks=5
        data_size=args.seq_len
    if args.dataset.startswith('ETTm'):
        num_tasks=5
        data_size=args.seq_len
    if args.dataset == 'ILI':
        num_tasks=5
        data_size=args.seq_len


    # Defining dataloaders for each domain
    dataloaders = dataset_preparation(args, num_tasks)
    forecast_model = Model(args).to(device)
    rnn_unit = RNN(data_size, device, forecast_model, args).to(device)
    
    # Loss and optimizer
    optimizer = torch.optim.Adam(rnn_unit.parameters(), lr=args.learning_rate)
    
    starting_time = time.time()
    # Training
    Es, hiddens = [None], [None]
    for task_id, dataloader in enumerate(dataloaders[:-2]):
        E, hidden, rnn_unit = train(dataloader, optimizer, rnn_unit, args, task_id, Es[-1], hiddens[-1])
        Es.append(E)
        hiddens.append(hidden)
        log("========== Finished Task #{} ==========".format(task_id))
    ending_time = time.time()
    log(f"Training time: {ending_time - starting_time}")
    
    # Testing
    val_mse, val_mae = evaluation(dataloaders[-2], rnn_unit, args, Es[-1], hiddens[-1])
    test_mse, test_mae = evaluation(dataloaders[-1], rnn_unit, args, Es[-1], hiddens[-1])
    return val_mse, val_mae, test_mse, test_mae

def tune_exp(config, args):
    log(f"tuning trail for {config}")
    
    # configure hyper-param for the trail
    args.noise_dim = config["noise_dim"]
    args.latent_dim = config["latent_dim"]
    args.hidden_dim = config["hidden_dim"]
    args.learning_rate = config["learning_rate"]
    args.batch_size = config["batch_size"]
    args.epoches = config["epoches"]
    args.num_rnn_layer = config["num_rnn_layer"]
    args.beta = config["beta"]

    val_mse, val_mae, test_mse, test_mae = trail(args)
    
    return val_mse, val_mae, test_mse, test_mae

def main(args, num_tails=10):
    configs = []
    for i in range(num_tails):
        dim = np.random.choice([512, 256, 128, 768, 64, 1024])
        config = {
            "noise_dim": dim,
            "latent_dim": dim,
            "hidden_dim": dim,
            "learning_rate": np.random.choice([5e-3, 2.5e-3, 1e-3, 5e-4]),
            "batch_size": int(np.random.choice([64, 128, 256])),
            "epoches": int(np.random.choice([100, 120, 150, 200, 80, 50])),
            # "epoches": int(np.random.choice([10])),
            "num_rnn_layer": int(np.random.choice([10, 5, 12, 20])),
            "beta": np.random.choice([0.1, 0.33, 0.4, 0.67, 0.9])
        }
        configs.append(config)

        log(f"Generated config {i}: {config}")

    best_result_ = 1000
    best_result = {}
    best_config = None

    for config in configs:
        val_mse, val_mae, test_mse, test_mae = tune_exp(config, args)
        
        temp_result_ = val_mae + val_mse ** (0.5)
        if temp_result_ < best_result_:
            best_result_ = temp_result_
            best_config = config
            best_result = {"val_MSE": val_mse, "val_MAE": val_mae, "test_MSE": test_mse, "test_MAE": test_mae}

    log("Best trial config: {}".format(best_config))
    log("Best trial final validation MSE: {}".format(
        best_result["val_MSE"]))
    log("Best trial final validation MAE: {}".format(
        best_result["val_MAE"]))
    log("Retreived final TEST MSE: {}".format(
        best_result["test_MSE"]))
    log("Retreived final TEST MAE: {}".format(
        best_result["test_MAE"]))

    # tune_exp(best_config, args)


if __name__ == "__main__":
    main(args, num_tails=100)






