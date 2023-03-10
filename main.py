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

seed_everything(123, workers=True)
# set precision for rtx3090
torch.set_float32_matmul_precision('highest')


#print('Is GPU available? {}'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DomainGen_Graph")

datasets = ['ONP', 'Moons', 'MNIST', 'Elec2']
parser.add_argument("--dataset", default="Moons", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))

# Hyper-parameters
parser.add_argument("--noise_dim", default=512, type=int,
                    help="the dimension of the LSTM input noise.")
parser.add_argument("--num_rnn_layer", default=10, type=int,
                    help="the number of RNN hierarchical layers.")
parser.add_argument("--latent_dim", default=512, type=int,
                    help="the latent dimension of RNN variables.")
parser.add_argument("--hidden_dim", default=512, type=int,
                    help="the latent dimension of RNN variables.")
parser.add_argument("--noise_type", choices=["Gaussian", "Uniform"], default="Gaussian",
                    help="The noise type to feed into the generator.")

parser.add_argument("--num_workers", default=0, type=int,
                    help="the number of threads for loading data.")
parser.add_argument("--epoches", default=200, type=int,
                    help="the number of epoches for each task.")
parser.add_argument("--batch_size", default=128, type=int,
                    help="the number of epoches for each task.")
parser.add_argument("--learning_rate", default=1e-3, type=float,
                    help="the unified learning rate for each single task.")

parser.add_argument("--is_test", default=True, type=bool,
                    help="if this is a testing period.")

# forecasting task
parser.add_argument('--seq_len', type=int, default=192, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels

args = parser.parse_args()
# if args.seq_len > 192:
#     args.noise_dim = int(args.pred_len/ 92 * args.noise_dim)
#     args.noise_dim = max(args.noise_dim, 2048)
#     args.latent_dim = args.noise_dim
#     args.hidden_dim = args.noise_dim
print(f"dim: {args.noise_dim}")

# setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
log_dir = f"log-{args.dataset}"
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
    objective = torch.nn.SmoothL1Loss(beta=.4)
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
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    log("Testing MAE is {}".format(mae))
    log("Testing MSE is {}".format(mse))


def main(args):
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
        print("========== Finished Task #{} ==========".format(task_id))
    ending_time = time.time()
    print("Training time:", ending_time - starting_time)
    
    # Testing
    evaluation(dataloaders[-2], rnn_unit, args, Es[-1], hiddens[-1])
    evaluation(dataloaders[-1], rnn_unit, args, Es[-1], hiddens[-1])
        

if __name__ == "__main__":
    print("Start Training...")
    
    # Initialize the time
    #starting_time = time.time()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    main(args)







