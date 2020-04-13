import copy
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import Generator, Encoder, EncoderLayer, Decoder, DecoderLayer, \
    MultiHeadedAttention, Embeddings, PositionwiseFeedForward, \
    PositionalEncoding, subsequent_mask, decode_sentence


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, cnn3d):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.cnn3d = cnn3d

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        #print("starting forward pass")
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask):
        #return self.encoder(self.src_embed(src), src_mask)
        video_features = []
        for s in src:
            clip_features = self.cnn3d(s.to(device))
            video_features.append(clip_features)
        #print("clip_features", clip_features.size())
        #print("video_features", len(video_features))
        features = torch.cat(video_features)
        return self.encoder(features, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = None#(torch.sum(src.view(src.size()[0],src.size()[1], -1),dim=-1) != 0).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def make_model(cnn3d, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8,
               dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
        cnn3d)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.named_parameters():
        if not p[0].startswith("cnn3d") and p[1].requires_grad and p[1].dim() > 1:
            nn.init.xavier_uniform_(p[1])

    return model


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        src, trg = batch
        batch = Batch(src, trg)
        out = model.forward(batch.src,
                            batch.trg.to(device),
                            batch.src_mask,#.to(device),
                            batch.trg_mask.to(device))
        loss = loss_compute(out.to(device), batch.trg_y.to(device), batch.ntokens.to(device))
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1, dtype=torch.int64).fill_(start_symbol)
    for i in range(max_len-1):
        out = model.decode(memory.to(device), src_mask,
                           Variable(ys).to(device),
                           Variable(subsequent_mask(ys.size(1))
                                    .type(torch.LongTensor)).to(device))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        if next_word == 2:
          break
        ys = torch.cat([ys,
                        torch.ones(1, 1, dtype=torch.int64).fill_(next_word)], dim=1)
    return ys


def evaluate_model(model, loader, device, max_seq, dictionary):
    token_corpus = []
    references = []
    for i, batch in enumerate(loader):
        print(i)
        src, trg = batch
        batch = Batch(src, trg)
        full_pred = greedy_decode(model,
                                  batch.src,
                                  batch.src_mask,
                                  max_len=max_seq,
                                  start_symbol=1).squeeze(dim=0)

        sentence = decode_sentence(full_pred[1:], dictionary)
        token_corpus.append(sentence)
        references.append(trg)
    return token_corpus, references
