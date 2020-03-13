from utils import json2keypoints
import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import DataLoader as dl
import adapted_transformer as tf
import csv
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

test_set = dl.SNLT_Dataset(train = False)

batch_size = 5
src_vocab = 274
trg_vocab = len(test_set.dictionary)
trg_len = 29

test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)
test_iter = iter(test_loader)
test_sample = next(test_iter)
src, trg = test_sample
real_sample = tf.Batch(src,trg)


print('real source',real_sample.src.size(),
      '\nreal target', real_sample.trg.size(),
      '\nreal source masked', real_sample.src_mask.size(),
      '\nreal trg masked', real_sample.trg_mask.size(),
      '\nreal trg y', real_sample.trg_y.size(),
      '\nreal ntokens', real_sample.ntokens)


criterion = tf.LabelSmoothing(size=trg_vocab, padding_idx=0, smoothing=0.0)
model = tf.make_model(src_vocab, trg_vocab, d_model=274, N=1, h=2)
model_opt = tf.NoamOpt(274, 1, 400, #i removed model.src_embed[0].d_model as first argument
            optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
#Encoding
encoded = model.encode(real_sample.src, real_sample.src_mask)
print('After going through encoder', encoded.size())
#Decoding
decoded = model.decode(encoded, real_sample.src_mask, real_sample.trg, real_sample.trg_mask)
print('After going through decoder', decoded.size())
#Forward full model
out = model.forward(real_sample.src, real_sample.trg, real_sample.src_mask, real_sample.trg_mask)
print('Full model forward', out.size())
pred = model.generator(out)
print('Generation step', pred.size())
SLC = tf.SimpleLossCompute(model.generator, criterion, model_opt)
loss = SLC(out, real_sample.trg_y, real_sample.ntokens)
print('Loss',loss)

losses = []
for epoch in range(5):
    print('Starting epoch:', epoch)
    model.train()
    loss = tf.run_epoch(test_loader, model, tf.SimpleLossCompute(model.generator, criterion, model_opt))
    losses.append(loss)
    '''
    model.eval()
    print(run_epoch(valid_loader, model,
          SimpleLossCompute(model.generator, criterion, None), epoch))
    '''

plt.plot(losses)
plt.ylabel('Train loss')
plt.show()
