import argparse
import math
import time
import dill as pickle
from tqdm import tqdm
import numpy as np
import random
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TextclassificationDataset

import model.Constants as Constants
from model.Models import BERT_like
# from model.Optim import ScheduledOptim