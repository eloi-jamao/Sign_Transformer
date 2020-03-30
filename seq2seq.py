import torch.nn as nn
import model
# import transformer
from transformer import *
import feature_extraction.resnets_3d.resnet as rn


# TODO
class Seq2Seq(nn.Module):
    def __init__(self, opts):
        super(Seq2Seq, self).__init__()

        self.d_model = opts.d_model
        resnet = rn.resnet34(num_classes=opts.n_classes, shortcut_type=opts.resnet_shortcut,
                             sample_size=opts.sample_size,
                             sample_duration=opts.sample_duration)
        self.extractor = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1,
                                       resnet.layer2, resnet.layer3, resnet.layer4)
        self.rl = nn.ReLU(inplace=True)
        c = copy.deepcopy
        attn = MultiHeadedAttention(opts.att_heads, opts.d_model)
        ff = PositionwiseFeedForward(opts.d_model, opts.d_ff, opts.dropout)
        position = PositionalEncoding(opts.d_model, opts.dropout)
        #
        self.encoder = Encoder(EncoderLayer(512, c(attn), c(ff), opts.dropout), opts.n_blocks)
        self.decoder = Decoder(DecoderLayer(opts.d_model, c(attn), c(attn), c(ff), opts.dropout), opts.n_blocks)
        # self.src_embed = nn.Sequential(Embeddings(opts.d_model, opts.src_vocab), c(position))#TODO
        self.tgt_embed = nn.Sequential(Embeddings(opts.d_model, opts.trg_vocab), c(position))
        self.generator = Generator(opts.d_model, opts.trg_vocab)
        # self.transformer = EncoderDecoder(self.encoder, self.decoder, self.src_embed, self.tgt_embed, self.generator)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        features = []
        for frames_group in src:
            feature = self.extractor(frames_group)
            features.concat(feature)
        ft = torch.stack(features)
        ft = ft.view(ft.size(0), -1)
        # y = self.encoder(self.src_embed(src), src_mask)
        y = self.encoder(ft, src_mask)
        y = self.decoder(self.tgt_embed(tgt), y, src_mask, tgt_mask)

        return y



