import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from torchtext import datasets
import random
import numpy as np
import pandas as pd
from collections import Counter
from typing import List
from kiwipiepy import Kiwi

kiwi = Kiwi()

def most_frequent_number(lst):
    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][0]

df = pd.read_excel('data/dataset.xlsx')
token_count = []
for i in range(len(df)):
  token_count.append(len(kiwi.tokenize(df['Sentence'][i], normalize_coda=False, split_complex=False, blocklist=None)))
print(most_frequent_number(token_count))
