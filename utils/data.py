import numpy as np
import torch
from torch.utils.data import Dataset


class PatientDataset(Dataset):
    def __init__(self, data_file, survival_file, keep_ids=None, transforms: tuple=None, no_time_no_death=True):
        """
        Arguments:
            data_file (string): Path to the health data csv file with annotations.
            survival_file (string): Path to csv with ground truth survival functions.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            keep_ids: use to define ids to keep for train/test split.
        """
        data = np.genfromtxt(data_file, delimiter=",")[1:]
        if no_time_no_death:
            data = data[:,:-2]
        true = np.genfromtxt(survival_file, delimiter=",")[1:].T
        true_days = true[0]
        true_data = true[1:]
        interp_days = np.arange(1, np.max(true_days) + 1)

        # interpolate daly values
        true_interp = torch.tensor(
            np.asarray([np.interp(interp_days, true_days, fn) for fn in true_data])
        )

        self.max_vals = torch.tensor(np.max(data, axis=0))
        self.min_vals = torch.tensor(np.min(data, axis=0))
        if transforms is None:
            self.transform = lambda x: 2 * ((x - self.min_vals) / (self.max_vals - self.min_vals)) - 1
            self.inv_transform = lambda x: (x + 1) / 2 * (self.max_vals - self.min_vals) + self.min_vals
        else:
            self.transform = transforms[0]
            self.inv_transform = transforms[1]

        # select with ids
        if keep_ids is not None:
            data_dims = len(data.shape)
            true_dims = len(true_interp.shape)
            data = data[keep_ids]
            true_interp = true_interp[keep_ids]
            if len(data.shape) < data_dims:
                data = data[None, :]
            if len(true_interp.shape) < true_dims:
                true_interp = true_interp[None, :]

        # normalize from -1 to 1
        self.x = torch.tensor(data)
        self.y = true_interp

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = self.transform(self.x[index]).type(torch.FloatTensor)
        y = self.y[index].type(torch.FloatTensor)
        return x, y
