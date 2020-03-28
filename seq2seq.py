import torch.nn as nn
import model
# import transformer
from transformer import *
import feature_extraction.resnets_3d.resnet as rn


# TODO
class Seq2Seq(nn.Module):
    def __init__(self, opts):
        super(Seq2Seq, self).__init__()

        resnet = rn.resnet34(num_classes=opts.n_classes, shortcut_type=opts.resnet_shortcut,
                             sample_size=opts.sample_size,
                             sample_duration=opts.sample_duration)
        self.extractor = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1,
                                       resnet.layer2, resnet.layer3, resnet.layer4)
        self.rl = nn.ReLU(inplace=True)
        # src_vocab, trg_vocab, N=N_blocks, d_model=d_model, h= att_heads
        # src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
        c = copy.deepcopy
        attn = MultiHeadedAttention(opts.att_heads, opts.d_model)
        ff = PositionwiseFeedForward(opts.d_model, opts.d_ff, opts.dropout)
        position = PositionalEncoding(opts.d_model, opts.dropout)
        #
        self.encoder = Encoder(EncoderLayer(opts.d_model, c(attn), c(ff), opts.dropout), opts.n_blocks)
        self.decoder = Decoder(DecoderLayer(opts.d_model, c(attn), c(attn), c(ff), opts.dropout), opts.n_blocks)
        self.src_embed = nn.Sequential(Embeddings(opts.d_model, opts.src_vocab), c(position))#TODO
        self.tgt_embed = nn.Sequential(Embeddings(opts.d_model, opts.trg_vocab), c(position))
        self.generator = Generator(opts.d_model, opts.trg_vocab)
        self.transformer = EncoderDecoder(self.encoder, self.decoder, self.src_embed, self.tgt_embed, self.generator)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # y = self.resnet(x)
        # y = y.view(y.size(0), -1)
        # y = self.rl(y)
        # # y = self.linear(y)
        y = self.transformer(x)
        return y

# class TransformerOpts(object):
#     ntoken = None
#     ninp = None
#     nhead = None
#     nhid = None
#     nlayers = None
#     vocab_size = None
#     max_seq = None
#     dropout = None
#
#     def __init__(self, **kwargs):
#         self.ntoken = kwargs.get('ntoken', None)
#         self.ninp = kwargs.get('ninp', None)
#         self.nhead = kwargs.get('nhead', None)
#         self.nhid = kwargs.get('nhid', None)
#         self.nlayers = kwargs.get('nlayers', None)
#         self.vocab_size = kwargs.get('vocab_size', None)
#         self.max_seq = kwargs.get('max_seq', None)
#         self.dropout = kwargs.get('dropout', None)
#
#
# class ResnetOpts(object):
#     n_classes = None
#     resnet_shortcut = None
#     sample_size = None
#     sample_duration = None
#
#     def __init__(self, **kwargs):
#         self.n_classes = kwargs.get('n_classes', None)
#         self.resnet_shortcut = kwargs.get('resnet_shortcut', None)
#         self.sample_size = kwargs.get('sample_size', None)
#         self.sample_duration = kwargs.get('sample_duration', None)
