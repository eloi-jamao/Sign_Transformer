import model
import utils
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import DataLoader.py as dl

dataset = dl.SNLT_Dataset(train = False)
dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

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

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
