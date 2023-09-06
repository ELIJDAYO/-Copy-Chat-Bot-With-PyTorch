import json
from nltk_utils import tokenizer, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenizer(pattern)
        # we don't want arr of arr
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', '.']
# stem each w for each w in  all words and exclude ignore words
all_words = [stem(w) for w in all_words if w not in ignore_words]
# set removes duplicate
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(tags)

X_train = []
y_train = []

for(pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) # CrossEntropyLoss instead of one-hot-encoding

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

# hyper parameters
batch_size = 8
dataset = ChatDataset()
# num_workers as multiprocessing task
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)