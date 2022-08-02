import os

import pandas as pd

import torch
from torch.utils.data import Dataset

class CustomTextDataset(Dataset):
    def __init__(self, DATA_PATH):
        self.labels = pd.read_csv('DATA_PATH')
        self.text = 