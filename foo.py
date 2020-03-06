import model
import utils
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import DataLoader as dl

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

test_set = dl.SNLT_Dataset(train = False)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = True)

num_keypoints = 274
emb_size = 200
vocab_size = len(test_set.dictionary)
""""In our project ntoken = number of keypoints because that is our input, not words"""

model = model.TransformerModel(
                               ntoken = num_keypoints,
                               ninp = emb_size,
                               nhead = 2,
                               nhid = 100,
                               vocab_size = vocab_size,
                               nlayers = 2,
                               dropout = 0.2
                               )

criterion = nn.NLLLoss()
opt = optim.Adam(model.parameters())

test_iter = iter(test_loader)
data, target = next(test_iter)
output = model(data)
print(output.shape, target.shape)

output = output[0][:target.shape[-1]].unsqueeze(dim=0)
target = nn.functional.one_hot(target, num_classes=vocab_size)
print(output.shape, target.shape)

loss = criterion(output.contiguous().view(-1, output.size(-1)), target.contiguous().view(-1))
print(loss)
