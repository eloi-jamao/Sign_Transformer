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



def check_model(model, val_set):
    sample = val_set[0][0]
    sample.unsqueeze(dim=0)
    print(sample.shape)
    output = model(sample)
    output.squeeze(dim=0)
    print(output.shape)    
    values, indices = torch.max(output,0)
    prediction = decode_output(indices, val_loader.dictionary)
    target = val_loader[0][1]
    GT = decode_output(target, val_loader.dictionary)
    print(f'Sentence predicted:{prediction}\nGT:{GT}')

def decode_output(x, dictionary):
    '''x is a list with the indices of the words'''
    sentence = []
    for i in x:
        if i == 2:
            return sentence
        else:
            sentence.append(dictionary.idx2word[i])
    return sentence

def train(train_loader, model, criterion, optimizer, epoch, device):

    # switch to train mode
    model.train()

    for i, (data, target) in enumerate(train_loader):
        # reset gradients
        optimizer.zero_grad()
        # move data to gpu
        data = data.to(device)
        target = target.to(device)
        # compute output
        output = model(data)
        # loss
        output, target = output[0][:target.shape[-1]], target.squeeze(dim=0)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f'TRAIN [{i+1}/{len(train_loader)}] Loss {loss}')

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

epochs = 3

for epoch in range(epochs):
    # train for one epoch
    print('Epoch:', epoch)
    check_model(model, test_set)
    train(test_loader, model, criterion, opt, epoch, device)

    # evaluate on validation set
    #acc1 = validate(val_loader, model, criterion, device)
