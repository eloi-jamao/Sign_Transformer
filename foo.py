import model
import utils
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import DataLoader.py as dl

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


num_keypoints = 274
emb_size = 200
""""In our project ntoken = number of keypoints because that is our input, not words"""

model = model.TransformerModel(
                               ntoken = num_keypoints,
                               ninp = emb_size,
                               nhead = 2,
                               nhid = 100,
                               nlayers = 2,
                               dropout = 0.2
                               )

criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters())

test_set = dl.SNLT_Dataset(train = False)
test_loader = DataLoader(dataset, batch_size = 1, shuffle = True)
test_iter = iter(test_loader)
data, target = next(test_iter)
output = model(data)
print(output.shape)
