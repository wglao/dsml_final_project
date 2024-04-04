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
    