from utils import json2keypoints
import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import DataLoader as dl
import adapted_transformer as tf
import matplotlib.pyplot as plt
from torch.autograd import Variable

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

frames_path = './../Sign_Transformer1/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px' #change to your paths
csv_path = './../Sign_Transformer1/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'

test_set = dl.SNLT_Dataset(split = 'test', frames_path = frames_path, csv_path = csv_path, gloss = False)


batch_size = 5
proj_size = 4096
trg_vocab = len(test_set.dictionary)
in_len = 475
out_len = 35

test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)
test_iter = iter(test_loader)
test_sample = next(test_iter)
src, trg = test_sample
#src = utils.load_images(src)
real_sample = tf.Batch(src,trg)


criterion = tf.LabelSmoothing(size=trg_vocab, padding_idx=0, smoothing=0.0)
model = tf.make_model(proj_size, trg_vocab, d_model=274, N=2, h=2)
model_opt = tf.NoamOpt(274, 1, 400, #i removed model.src_embed[0].d_model as first argument
            optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
if False: #Change to True/False to see/unsee stats
    print('real source',real_sample.src.size(),
          '\nreal target', real_sample.trg.size(),
          '\nreal source masked', real_sample.src_mask.size(),
          '\nreal trg masked', real_sample.trg_mask.size(),
          '\nreal trg y', real_sample.trg_y.size(),
          '\nreal ntokens', real_sample.ntokens)
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

def check_result(dataset, model, in_len = 475, max_len=35, start_symbol=0):
    src, trg = dataset[0]
    model.eval()
    src = Variable(src.unsqueeze(dim=0))
    src_mask = Variable(torch.ones(1, 1, in_len))
    pred = tf.greedy_decode(model, src, src_mask, max_len=out_len, start_symbol=start_symbol)
    trg_sent =[dataset.dictionary.idx2word[i] for i in trg]
    pred_sent = [dataset.dictionary.idx2word[i] for i in pred[0]]
    return trg_sent, pred_sent

losses = []
for epoch in range(0):
    print('Starting epoch:', epoch)
    model.train()
    loss = tf.run_epoch(test_loader, model, tf.SimpleLossCompute(model.generator, criterion, model_opt))
    losses.append(loss)

    trg_sent, pred_sent = check_result(test_set, model)
    print(trg_sent,'\n',pred_sent)
    '''
    model.eval()
    print(run_epoch(valid_loader, model,
          SimpleLossCompute(model.generator, criterion, None), epoch))
    '''
show_plot = False
if show_plot:
    plt.plot(losses)
    plt.ylabel('Train loss')
    plt.show()
