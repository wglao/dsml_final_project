import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=0, type=int)

args = parser.parse_args()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PatientDataset(Dataset):
    def __init__(self, data_file, survival_file, keep_ids=None):
        """
        Arguments:
            data_file (string): Path to the health data csv file with annotations.
            survival_file (string): Path to csv with ground truth survival functions.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            keep_ids: use to define ids to keep for train/test split.
        """
        data = np.genfromtxt(data_file, delimiter=",")[1:]
        true = np.genfromtxt(survival_file, delimiter=",")[1:].T

        max_vals = np.max(data, axis=0)
        min_vals = np.min(data, axis=0)
        self.transform = lambda x: 2 * ((x - min_vals) / (max_vals - min_vals)) - 1

        # select with ids
        if keep_ids is not None:
            data_dims = len(data.shape)
            true_dims = len(true.shape)
            data = data[keep_ids]
            true = true[keep_ids]
            if len(data.shape) < data_dims:
                data = data[None, :]
            if len(true.shape) < true_dims:
                true = true[None, :]

        # normalize from -1 to 1
        self.x = data
        self.y = true

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.tensor(self.transform(x[index]))
        y = torch.tensor(self.y[index])
        return x, y
