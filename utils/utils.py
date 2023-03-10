# -*- coding: utf-8 -*-

import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom


def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.
    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".
    Returns
    -------
    noise tensor
    """

    if type == "Gaussian":
        noise = Variable(torch.randn(shape))
    elif type == 'Uniform':
        noise = Variable(torch.randn(shape).uniform_(-1, 1))
    else:
        raise Exception("ERROR: Noise type {} not supported".format(type))
    return noise


def dataset_preparation(args, num_tasks=10):
    if args.dataset.startswith("ETT"):
        data_root = "../../../tsa/datasets/ETT-data"
        data_path = f"{args.dataset}.csv"
        dataloaders = []

        if data_path.startswith("ETTh"):
            dataset = Dataset_ETT_hour
        if data_path.startswith("ETTm"):
            dataset = Dataset_ETT_minute
    else:
        data_root = f"../../../tsa/datasets/"
        if args.dataset == "ILI":
            data_path = "illness/national_illness.csv"

        dataset = Dataset_Custom
    dataloaders = []
    for taskid in range(num_tasks):
        domain_dataset = dataset(
            root_path=data_root,
            data_path=data_path,
            size=[args.seq_len, args.label_len, args.pred_len],
            features='M',
            taskid=taskid
        )
        temp_dataloader = DataLoader(domain_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers)
        dataloaders.append(temp_dataloader)
    return dataloaders
